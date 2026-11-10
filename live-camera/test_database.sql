-- Integration checks against an initialized project database. Fixtures roll back.
\set ON_ERROR_STOP on
BEGIN;
DO $$
DECLARE
    sid uuid := gen_random_uuid();
    other_sid uuid := gen_random_uuid();
    event uuid := gen_random_uuid();
    voice uuid := gen_random_uuid();
    found integer;
BEGIN
    IF current_database() <> 'live_feed_analysis' THEN
        RAISE EXCEPTION 'Database checks require live_feed_analysis';
    END IF;
    INSERT INTO sessions(id,question,room_name,publisher_identity)
        VALUES(sid,'Observe continuously','test-'||sid,'test-publisher'),
              (other_sid,'Another session','test-'||other_sid,'other-publisher');
    INSERT INTO vision_observations(session_id,event_id,start_seconds,end_seconds,state,description,sample_times)
        VALUES(sid,event,1,2,'response','Cup beside laptop',ARRAY[1,1.25,1.5,1.75]);
    SELECT count(*) INTO found FROM vision_observations
        WHERE session_id=sid AND search_text @@ plainto_tsquery('simple','cup laptop');
    IF found <> 1 THEN RAISE EXCEPTION 'Full-text lookup failed'; END IF;
    INSERT INTO vision_evidence(session_id,event_id,frame_index,sample_time_seconds,jpeg)
        VALUES(sid,event,0,1,decode('ffd8ffd9','hex'));
    INSERT INTO voice_sessions(id,session_id,model) VALUES(voice,sid,'test-voice');
    INSERT INTO voice_transcript_segments(session_id,voice_session_id,provider_event_id,role,text,start_ms,end_ms)
        VALUES(sid,voice,'test-event','user','Where is the cup?',1000,2000);
    INSERT INTO model_calls(session_id,observation_event_id,component,operation,status,completed_at)
        VALUES(sid,event,'memory_tool','search_vision_memory','complete',now());
    INSERT INTO debug_events(session_id,observation_event_id,level,component,event_type,message)
        VALUES(sid,event,'info','test','observation.received','Test event');

    BEGIN
        INSERT INTO voice_transcript_segments(session_id,voice_session_id,provider_event_id,role,text,start_ms,end_ms)
            VALUES(other_sid,voice,'wrong-session','user','Must fail',0,1);
        RAISE EXCEPTION 'Cross-session transcript was accepted';
    EXCEPTION WHEN foreign_key_violation THEN NULL;
    END;
    BEGIN
        INSERT INTO vision_evidence(session_id,event_id,frame_index,sample_time_seconds,jpeg)
            VALUES(other_sid,event,0,1,decode('ff','hex'));
        RAISE EXCEPTION 'Cross-session evidence was accepted';
    EXCEPTION WHEN foreign_key_violation THEN NULL;
    END;
    BEGIN
        INSERT INTO vision_observations(session_id,event_id,start_seconds,end_seconds,state)
            VALUES(sid,gen_random_uuid(),2,1,'silence');
        RAISE EXCEPTION 'Reversed interval was accepted';
    EXCEPTION WHEN check_violation THEN NULL;
    END;
    BEGIN
        INSERT INTO vision_observations(session_id,event_id,start_seconds,end_seconds,state,sample_times)
            VALUES(sid,gen_random_uuid(),1,2,'silence',ARRAY[3.0]);
        RAISE EXCEPTION 'Out-of-range sample was accepted';
    EXCEPTION WHEN check_violation THEN NULL;
    END;
    BEGIN
        INSERT INTO vision_observations(session_id,event_id,start_seconds,end_seconds,state)
            VALUES(sid,gen_random_uuid(),0,'NaN'::double precision,'silence');
        RAISE EXCEPTION 'NaN time was accepted';
    EXCEPTION WHEN check_violation THEN NULL;
    END;
    BEGIN
        INSERT INTO vision_observations(session_id,event_id,start_seconds,end_seconds,state)
            VALUES(sid,event,1,2,'response');
        RAISE EXCEPTION 'Duplicate event was accepted';
    EXCEPTION WHEN unique_violation THEN NULL;
    END;
    BEGIN
        UPDATE sessions SET status='completed' WHERE id=sid;
        RAISE EXCEPTION 'Completed session without end time was accepted';
    EXCEPTION WHEN check_violation THEN NULL;
    END;
    UPDATE sessions SET status='completed',ended_at=now() WHERE id=sid;
    -- Evidence can expire while its descriptive memory remains searchable.
    DELETE FROM vision_evidence WHERE session_id=sid;
    IF NOT EXISTS (SELECT 1 FROM vision_observations WHERE session_id=sid) THEN
        RAISE EXCEPTION 'Pruning evidence deleted memory';
    END IF;
    DELETE FROM sessions WHERE id=sid;
    IF EXISTS (SELECT 1 FROM vision_observations WHERE session_id=sid)
        OR EXISTS (SELECT 1 FROM voice_transcript_segments WHERE session_id=sid)
        OR EXISTS (SELECT 1 FROM model_calls WHERE session_id=sid)
        OR EXISTS (SELECT 1 FROM debug_events WHERE session_id=sid) THEN
        RAISE EXCEPTION 'Session cleanup left orphan records';
    END IF;
    RAISE NOTICE 'PASS: persistence, search, session isolation, timing, deduplication, lifecycle, evidence pruning and cascade cleanup';
END $$;
ROLLBACK;
