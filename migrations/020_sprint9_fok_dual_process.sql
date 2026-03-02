-- Migration 020: Sprint 9 — FOK Dual-Process columns
-- Metcalfe et al. 1993, Koriat 1993: two-signal FOK model
-- Adds familiarity_score, accessibility_score, fok_process to calibration log
-- When: 2026-02-28

ALTER TABLE fok_calibration_log ADD COLUMN familiarity_score REAL DEFAULT NULL;
ALTER TABLE fok_calibration_log ADD COLUMN accessibility_score REAL DEFAULT NULL;
ALTER TABLE fok_calibration_log ADD COLUMN fok_process TEXT DEFAULT 'single_process';
