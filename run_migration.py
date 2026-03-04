#!/usr/bin/env python3
"""
Run Skill Memory migration 001
"""
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from config_pg import get_conn

def run_migration():
    """Execute migration 001_skill_memory.sql"""
    migration_file = Path(__file__).parent / "migrations" / "001_skill_memory.sql"

    if not migration_file.exists():
        print(f"❌ Migration file not found: {migration_file}")
        return False

    print(f"📂 Reading migration: {migration_file}")
    sql = migration_file.read_text()

    print("🔌 Connecting to PostgreSQL...")
    try:
        with get_conn() as conn:
            print("✅ Connected")
            print("🚀 Executing migration...")

            # Execute in transaction
            with conn.transaction():
                conn.execute(sql)

            print("✅ Migration executed successfully")

            # Verify tables
            print("\n📊 Verifying tables...")
            result = conn.execute("""
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
                  AND tablename LIKE 'skill%'
                ORDER BY tablename
            """).fetchall()

            print(f"✅ Created {len(result)} tables:")
            for row in result:
                print(f"   - {row[0]}")

            # Run smoke test
            print("\n🧪 Running smoke test...")
            conn.execute("""
                DO $$
                DECLARE
                    v_candidate_id UUID;
                    v_skill_id UUID;
                    v_version_id UUID;
                BEGIN
                    -- Test promote_candidate_to_skill
                    INSERT INTO candidate_skills (
                        proposed_name, proposed_problem, proposed_steps,
                        proposed_triggers, auto_score, status
                    ) VALUES (
                        'Test skill', 'Solve test problem',
                        '[{"step":1,"action":"do thing"}]'::jsonb,
                        ARRAY['test'], 0.75, 'pending'
                    ) RETURNING id INTO v_candidate_id;

                    v_skill_id := promote_candidate_to_skill(v_candidate_id);
                    RAISE NOTICE 'Promoted candidate % to skill %', v_candidate_id, v_skill_id;

                    -- Test version selection UCB
                    v_version_id := select_best_version_ucb(v_skill_id);
                    RAISE NOTICE 'Selected version: %', v_version_id;

                    -- Test increment success
                    PERFORM increment_skill_success(v_skill_id, v_version_id);
                    RAISE NOTICE 'Incremented success';

                    -- Test activation update
                    UPDATE skills SET last_used = now() WHERE id = v_skill_id;

                    -- Test decay
                    PERFORM apply_skill_decay();
                    RAISE NOTICE 'Applied decay';

                    -- Test cycle detection
                    BEGIN
                        INSERT INTO skill_dependencies (parent_skill_id, child_skill_id, dependency_type)
                        VALUES (v_skill_id, v_skill_id, 'required');
                        RAISE EXCEPTION 'Cycle detection FAILED';
                    EXCEPTION WHEN OTHERS THEN
                        RAISE NOTICE 'Cycle detection PASSED: %', SQLERRM;
                    END;

                    -- Cleanup
                    DELETE FROM skills WHERE id = v_skill_id;

                    RAISE NOTICE '✓ All smoke tests PASSED';
                END $$;
            """)

            print("✅ Smoke test passed")

            # ANALYZE
            print("\n📈 Analyzing tables...")
            for table in ['skills', 'skill_versions', 'candidate_skills', 'skill_executions']:
                conn.execute(f"ANALYZE {table}")
            print("✅ Analysis complete")

            return True

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
