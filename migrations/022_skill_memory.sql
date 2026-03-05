-- ============================================
-- SKILL MEMORY - Production Ready Schema v5
-- Migration 001
-- Date: 2026-03-03
-- ============================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================
-- CORE: Skill canónico
-- ============================================
CREATE TABLE IF NOT EXISTS skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    skill_name TEXT NOT NULL,
    canonical_problem TEXT NOT NULL,
    context_triggers TEXT[] DEFAULT '{}',
    tools_used TEXT[] DEFAULT '{}',
    embedding vector(1536),

    -- Performance tracking
    success_count INT NOT NULL DEFAULT 0 CHECK (success_count >= 0),
    failure_count INT NOT NULL DEFAULT 0 CHECK (failure_count >= 0),

    -- Confidence con prior bayesiano
    confidence_score NUMERIC(5,4) DEFAULT 0.5,

    -- ACT-R activation y decay
    storage_strength NUMERIC(5,4) DEFAULT 1.0 CHECK (storage_strength BETWEEN 0 AND 1),
    retrieval_strength NUMERIC(5,4) DEFAULT 1.0 CHECK (retrieval_strength BETWEEN 0 AND 1),
    access_timestamps TIMESTAMPTZ[] DEFAULT '{}',
    base_level_activation NUMERIC(8,4) DEFAULT 0.0,

    last_used TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Full-text search ponderado
    search_vector tsvector,

    -- Lifecycle
    status TEXT CHECK (status IN ('active', 'deprecated', 'archived', 'superseded')) DEFAULT 'active',
    superseded_by_skill_id UUID REFERENCES skills(id),
    deprecation_reason TEXT,
    deprecated_at TIMESTAMPTZ,

    CONSTRAINT status_superseded_consistency CHECK (
        (status = 'superseded' AND superseded_by_skill_id IS NOT NULL) OR
        (status <> 'superseded' AND superseded_by_skill_id IS NULL)
    )
);

-- ============================================
-- VERSIONS: Diferentes approaches
-- ============================================
CREATE TABLE IF NOT EXISTS skill_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    skill_id UUID NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    version INT NOT NULL,
    strategy_name TEXT,

    -- Procedural knowledge
    solution_steps JSONB NOT NULL,
    steps_hash TEXT,

    preconditions JSONB DEFAULT '{}',
    postconditions JSONB DEFAULT '{}',

    -- Provenance
    source_episodes UUID[] DEFAULT '{}',

    -- Quality metrics
    quality_score NUMERIC(5,4) DEFAULT 0.5 CHECK (quality_score BETWEEN 0 AND 1),
    success_count INT NOT NULL DEFAULT 0 CHECK (success_count >= 0),
    failure_count INT NOT NULL DEFAULT 0 CHECK (failure_count >= 0),
    avg_execution_time_ms INT,

    -- Context fit
    context_fit_score NUMERIC(5,4) DEFAULT 0.5 CHECK (context_fit_score BETWEEN 0 AND 1),

    -- Selection metrics para UCB/Thompson
    exploration_bonus NUMERIC(5,4) DEFAULT 0.0,
    times_selected INT DEFAULT 0,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used TIMESTAMPTZ,

    UNIQUE(skill_id, version),
    UNIQUE(skill_id, steps_hash)
);

-- ============================================
-- VALIDATION QUEUE
-- ============================================
CREATE TABLE IF NOT EXISTS candidate_skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    proposed_name TEXT NOT NULL,
    proposed_problem TEXT NOT NULL,
    proposed_steps JSONB NOT NULL,
    proposed_triggers TEXT[] DEFAULT '{}',
    proposed_tools TEXT[] DEFAULT '{}',

    embedding vector(1536),

    source_episode UUID,

    -- Validation signals
    outcome_success BOOLEAN,
    has_tests BOOLEAN DEFAULT false,
    user_confirmed BOOLEAN DEFAULT false,
    auto_score NUMERIC(5,4) CHECK (auto_score BETWEEN 0 AND 1),

    status TEXT CHECK (status IN ('pending', 'validated', 'rejected', 'merged')) DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_at TIMESTAMPTZ
);

-- ============================================
-- EXECUTION LOG
-- ============================================
CREATE TABLE IF NOT EXISTS skill_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    skill_id UUID NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    version_id UUID REFERENCES skill_versions(id) ON DELETE SET NULL,

    episode_id UUID,

    -- Context
    triggered_by TEXT,
    context_snapshot JSONB,

    -- Outcome
    success BOOLEAN,
    execution_time_ms INT,
    error_message TEXT,

    executed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================
-- SKILL COMPOSITION
-- ============================================
CREATE TABLE IF NOT EXISTS skill_dependencies (
    parent_skill_id UUID NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    child_skill_id UUID NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    dependency_type TEXT CHECK (dependency_type IN ('required', 'optional', 'alternative')) NOT NULL,
    execution_order INT,
    PRIMARY KEY (parent_skill_id, child_skill_id),
    CONSTRAINT no_self_dependency CHECK (parent_skill_id <> child_skill_id)
);

-- ============================================
-- SKILL TEMPLATES
-- ============================================
CREATE TABLE IF NOT EXISTS skill_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_name TEXT NOT NULL,
    abstract_pattern TEXT NOT NULL,
    pattern_signature JSONB,
    embedding vector(1536),
    concrete_skills UUID[] DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================
-- TRIGGERS
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_skills_updated_at ON skills;
CREATE TRIGGER update_skills_updated_at
    BEFORE UPDATE ON skills
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ACT-R activation update trigger
CREATE OR REPLACE FUNCTION update_skill_activation()
RETURNS TRIGGER AS $$
BEGIN
    -- Append timestamp a access_timestamps
    NEW.access_timestamps = array_append(
        COALESCE(NEW.access_timestamps, '{}'),
        now()
    );

    -- Keep only last 20 accesses
    IF array_length(NEW.access_timestamps, 1) > 20 THEN
        NEW.access_timestamps = NEW.access_timestamps[
            array_length(NEW.access_timestamps, 1) - 19 :
            array_length(NEW.access_timestamps, 1)
        ];
    END IF;

    -- Compute base-level activation (ACT-R formula)
    IF array_length(NEW.access_timestamps, 1) > 0 THEN
        NEW.base_level_activation = ln(
            (
                SELECT SUM(
                    POWER(
                        GREATEST(EXTRACT(EPOCH FROM (now() - ts))::NUMERIC / 86400.0, 0.0001),
                        -0.5
                    )
                )
                FROM unnest(NEW.access_timestamps) AS ts
            )
        );
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_skill_activation_trigger ON skills;
CREATE TRIGGER update_skill_activation_trigger
    BEFORE UPDATE OF last_used ON skills
    FOR EACH ROW
    EXECUTE FUNCTION update_skill_activation();

-- Update steps_hash trigger
CREATE OR REPLACE FUNCTION update_steps_hash()
RETURNS TRIGGER AS $$
BEGIN
    NEW.steps_hash = md5(NEW.solution_steps::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_steps_hash_trigger ON skill_versions;
CREATE TRIGGER update_steps_hash_trigger
    BEFORE INSERT OR UPDATE OF solution_steps ON skill_versions
    FOR EACH ROW
    EXECUTE FUNCTION update_steps_hash();

-- Update confidence_score trigger
CREATE OR REPLACE FUNCTION update_skill_confidence()
RETURNS TRIGGER AS $$
BEGIN
    NEW.confidence_score = CASE
        WHEN (NEW.success_count + NEW.failure_count) = 0 THEN 0.5
        WHEN (NEW.success_count + NEW.failure_count) < 3 THEN 0.5
        ELSE CAST(NEW.success_count AS NUMERIC) / (NEW.success_count + NEW.failure_count)
    END;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_skill_confidence_trigger ON skills;
CREATE TRIGGER update_skill_confidence_trigger
    BEFORE INSERT OR UPDATE OF success_count, failure_count ON skills
    FOR EACH ROW
    EXECUTE FUNCTION update_skill_confidence();

-- Update search_vector trigger
CREATE OR REPLACE FUNCTION update_skill_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector =
        setweight(to_tsvector('spanish', coalesce(NEW.skill_name, '')), 'A') ||
        setweight(to_tsvector('spanish', coalesce(NEW.canonical_problem, '')), 'A') ||
        setweight(to_tsvector('spanish', array_to_string(NEW.context_triggers, ' ')), 'B');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_skill_search_vector_trigger ON skills;
CREATE TRIGGER update_skill_search_vector_trigger
    BEFORE INSERT OR UPDATE OF skill_name, canonical_problem, context_triggers ON skills
    FOR EACH ROW
    EXECUTE FUNCTION update_skill_search_vector();

-- ============================================
-- INDICES
-- ============================================

-- Vector (parcial)
CREATE INDEX IF NOT EXISTS idx_skills_embedding ON skills
    USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_candidates_embedding ON candidate_skills
    USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_templates_embedding ON skill_templates
    USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;

-- Full-text
CREATE INDEX IF NOT EXISTS idx_skills_search_vector ON skills
    USING GIN (search_vector);

-- Arrays
CREATE INDEX IF NOT EXISTS idx_skills_triggers ON skills USING GIN (context_triggers);
CREATE INDEX IF NOT EXISTS idx_skills_tools ON skills USING GIN (tools_used);

-- Performance
CREATE INDEX IF NOT EXISTS idx_skills_confidence ON skills (confidence_score DESC) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_skills_activation ON skills (base_level_activation DESC) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_skills_last_used ON skills (last_used DESC NULLS LAST) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_skills_status ON skills (status);

-- Versions
CREATE INDEX IF NOT EXISTS idx_versions_quality ON skill_versions (quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_versions_skill ON skill_versions (skill_id, version);
CREATE INDEX IF NOT EXISTS idx_versions_selection ON skill_versions (times_selected DESC);

-- Executions
CREATE INDEX IF NOT EXISTS idx_executions_skill ON skill_executions (skill_id, executed_at DESC);
CREATE INDEX IF NOT EXISTS idx_executions_episode ON skill_executions (episode_id) WHERE episode_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_executions_success ON skill_executions (success, executed_at DESC);

-- Candidates
CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidate_skills (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_candidates_score ON candidate_skills (auto_score DESC) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_candidates_episode ON candidate_skills (source_episode) WHERE source_episode IS NOT NULL;

-- ============================================
-- VIEWS
-- ============================================

CREATE OR REPLACE VIEW active_skills_with_stats AS
SELECT
    s.id,
    s.skill_name,
    s.canonical_problem,
    s.confidence_score,
    s.base_level_activation,
    s.success_count,
    s.failure_count,
    s.last_used,
    COUNT(DISTINCT sv.id) as version_count,
    MAX(sv.quality_score) as best_version_quality,
    COUNT(DISTINCT se.id) as total_executions,
    COALESCE(
        SUM(CASE WHEN se.success THEN 1 ELSE 0 END)::NUMERIC /
        NULLIF(COUNT(se.id), 0),
        s.confidence_score
    ) as actual_success_rate
FROM skills s
LEFT JOIN skill_versions sv ON sv.skill_id = s.id
LEFT JOIN skill_executions se ON se.skill_id = s.id
WHERE s.status = 'active'
GROUP BY s.id;

CREATE OR REPLACE VIEW candidates_ready_for_review AS
SELECT
    c.*,
    COUNT(s.id) as similar_skills_count
FROM candidate_skills c
LEFT JOIN skills s ON s.embedding <-> c.embedding < 0.15
WHERE c.status = 'pending'
  AND c.auto_score >= 0.65
  AND c.embedding IS NOT NULL
GROUP BY c.id
HAVING COUNT(s.id) = 0
ORDER BY c.auto_score DESC;

-- ============================================
-- FUNCTIONS
-- ============================================

-- promote_candidate_to_skill()
CREATE OR REPLACE FUNCTION promote_candidate_to_skill(
    p_candidate_id UUID,
    p_merge_threshold NUMERIC DEFAULT 0.85
) RETURNS UUID AS $$
DECLARE
    v_candidate RECORD;
    v_similar_skill RECORD;
    v_skill_id UUID;
    v_version_num INT;
BEGIN
    SELECT * INTO v_candidate FROM candidate_skills WHERE id = p_candidate_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Candidate % not found', p_candidate_id;
    END IF;

    IF v_candidate.status <> 'pending' THEN
        RAISE EXCEPTION 'Candidate % is not pending (status: %)', p_candidate_id, v_candidate.status;
    END IF;

    SELECT s.* INTO v_similar_skill
    FROM skills s
    WHERE s.embedding <-> v_candidate.embedding < (1 - p_merge_threshold)
      AND s.status = 'active'
    ORDER BY s.embedding <-> v_candidate.embedding
    LIMIT 1;

    IF FOUND THEN
        v_skill_id = v_similar_skill.id;

        SELECT COALESCE(MAX(version), 0) + 1 INTO v_version_num
        FROM skill_versions
        WHERE skill_id = v_skill_id;

        INSERT INTO skill_versions (
            skill_id, version, strategy_name, solution_steps,
            source_episodes, quality_score
        ) VALUES (
            v_skill_id,
            v_version_num,
            v_candidate.proposed_name,
            v_candidate.proposed_steps,
            ARRAY[v_candidate.source_episode],
            v_candidate.auto_score
        );

        UPDATE candidate_skills
        SET status = 'merged', reviewed_at = now()
        WHERE id = p_candidate_id;

    ELSE
        INSERT INTO skills (
            skill_name,
            canonical_problem,
            context_triggers,
            tools_used,
            embedding
        ) VALUES (
            v_candidate.proposed_name,
            v_candidate.proposed_problem,
            v_candidate.proposed_triggers,
            v_candidate.proposed_tools,
            v_candidate.embedding
        ) RETURNING id INTO v_skill_id;

        INSERT INTO skill_versions (
            skill_id, version, strategy_name, solution_steps,
            source_episodes, quality_score
        ) VALUES (
            v_skill_id,
            1,
            'Default approach',
            v_candidate.proposed_steps,
            ARRAY[v_candidate.source_episode],
            v_candidate.auto_score
        );

        UPDATE candidate_skills
        SET status = 'validated', reviewed_at = now()
        WHERE id = p_candidate_id;
    END IF;

    RETURN v_skill_id;
END;
$$ LANGUAGE plpgsql;

-- Increment helpers
CREATE OR REPLACE FUNCTION increment_skill_success(
    p_skill_id UUID,
    p_version_id UUID DEFAULT NULL
) RETURNS void AS $$
BEGIN
    UPDATE skills
    SET success_count = success_count + 1,
        last_used = now()
    WHERE id = p_skill_id;

    IF p_version_id IS NOT NULL THEN
        UPDATE skill_versions
        SET success_count = success_count + 1,
            last_used = now(),
            times_selected = times_selected + 1
        WHERE id = p_version_id;
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION increment_skill_failure(
    p_skill_id UUID,
    p_version_id UUID DEFAULT NULL
) RETURNS void AS $$
BEGIN
    UPDATE skills
    SET failure_count = failure_count + 1,
        last_used = now()
    WHERE id = p_skill_id;

    IF p_version_id IS NOT NULL THEN
        UPDATE skill_versions
        SET failure_count = failure_count + 1,
            last_used = now(),
            times_selected = times_selected + 1
        WHERE id = p_version_id;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Version selection con UCB
CREATE OR REPLACE FUNCTION select_best_version_ucb(
    p_skill_id UUID,
    p_exploration_factor NUMERIC DEFAULT 2.0
) RETURNS UUID AS $$
DECLARE
    v_total_selections INT;
    v_best_version RECORD;
BEGIN
    SELECT SUM(times_selected) INTO v_total_selections
    FROM skill_versions
    WHERE skill_id = p_skill_id;

    IF v_total_selections = 0 THEN
        v_total_selections = 1;
    END IF;

    SELECT *
    INTO v_best_version
    FROM skill_versions
    WHERE skill_id = p_skill_id
    ORDER BY (
        quality_score +
        p_exploration_factor * SQRT(
            LN(v_total_selections::NUMERIC) / GREATEST(times_selected, 1)
        )
    ) DESC
    LIMIT 1;

    RETURN v_best_version.id;
END;
$$ LANGUAGE plpgsql;

-- Cycle detection
CREATE OR REPLACE FUNCTION check_dependency_cycle(
    p_parent_id UUID,
    p_child_id UUID
) RETURNS BOOLEAN AS $$
DECLARE
    v_has_cycle BOOLEAN;
BEGIN
    WITH RECURSIVE dependency_chain AS (
        SELECT child_skill_id as skill_id
        FROM skill_dependencies
        WHERE parent_skill_id = p_child_id

        UNION

        SELECT sd.child_skill_id
        FROM skill_dependencies sd
        INNER JOIN dependency_chain dc ON dc.skill_id = sd.parent_skill_id
    )
    SELECT EXISTS (
        SELECT 1 FROM dependency_chain WHERE skill_id = p_parent_id
    ) INTO v_has_cycle;

    RETURN v_has_cycle;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION prevent_dependency_cycle()
RETURNS TRIGGER AS $$
BEGIN
    IF check_dependency_cycle(NEW.parent_skill_id, NEW.child_skill_id) THEN
        RAISE EXCEPTION 'Dependency cycle detected: % -> %', NEW.parent_skill_id, NEW.child_skill_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS prevent_dependency_cycle_trigger ON skill_dependencies;
CREATE TRIGGER prevent_dependency_cycle_trigger
    BEFORE INSERT OR UPDATE ON skill_dependencies
    FOR EACH ROW
    EXECUTE FUNCTION prevent_dependency_cycle();

-- Decay function
CREATE OR REPLACE FUNCTION apply_skill_decay(
    p_decay_rate NUMERIC DEFAULT 0.05
) RETURNS void AS $$
BEGIN
    UPDATE skills
    SET
        storage_strength = GREATEST(
            storage_strength * EXP(-p_decay_rate * EXTRACT(EPOCH FROM (now() - COALESCE(last_used, created_at))) / 86400.0),
            0.1
        ),
        retrieval_strength = GREATEST(
            retrieval_strength * EXP(-p_decay_rate * EXTRACT(EPOCH FROM (now() - COALESCE(last_used, created_at))) / 86400.0),
            0.1
        )
    WHERE status = 'active'
      AND last_used < now() - INTERVAL '7 days';
END;
$$ LANGUAGE plpgsql;

-- Comments
COMMENT ON TABLE skills IS 'Core skill repository with ACT-R activation tracking';
COMMENT ON COLUMN skills.base_level_activation IS 'ACT-R base-level activation: ln(sum(t_j^-0.5))';
COMMENT ON FUNCTION promote_candidate_to_skill IS 'Promotes validated candidate to skill or merges as version';
COMMENT ON FUNCTION select_best_version_ucb IS 'UCB algorithm for version selection with exploration/exploitation balance';
COMMENT ON FUNCTION check_dependency_cycle IS 'Detects cycles in skill dependency graph';
COMMENT ON FUNCTION apply_skill_decay IS 'Applies exponential decay to unused skills (run periodically)';
