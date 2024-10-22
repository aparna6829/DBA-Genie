import cx_Oracle
import psycopg2
import sys
import re

# lib_dir = r"C:\Users\Administrator\Downloads\instantclient-basic-windows.x64-23.4.0.24.05\instantclient_23_4"
# cx_Oracle.init_oracle_client(lib_dir=lib_dir)

def get_oracle_triggers(oracle_conn, schema_name):
    cursor = oracle_conn.cursor()
    try:
        cursor.execute("""
            SELECT 
                trigger_name,
                triggering_event,
                trigger_type,
                trigger_body,
                table_name
            FROM 
                all_triggers
            WHERE 
                owner = :schema_name
        """, schema_name=schema_name.upper())

        triggers = cursor.fetchall()
        return triggers
    except cx_Oracle.Error as e:
        print(f"Oracle Error: {e}")
        return []
    finally:
        cursor.close()

def convert_trigger_body(trigger_body):
    try:
        # Convert RAISE_APPLICATION_ERROR to RAISE EXCEPTION
        trigger_body = re.sub(r'RAISE_APPLICATION_ERROR\s*\(\s*-?\d+\s*,\s*([^)]+)\)', r'RAISE EXCEPTION \1', trigger_body)

        # Replace :NEW and :OLD with NEW and OLD
        trigger_body = re.sub(r':(\w+)\.', r'\1.', trigger_body)

        # Convert Oracle specific functions
        oracle_to_pg_functions = {
            'NVL': 'COALESCE',
            'SYSDATE': 'CURRENT_DATE',
            'SYSTIMESTAMP': 'CURRENT_TIMESTAMP',
            'TO_DATE': 'TO_DATE',  # Needs special handling
            'TO_CHAR': 'TO_CHAR',  # Might need adjustment based on format
        }

        for oracle_func, pg_func in oracle_to_pg_functions.items():
            trigger_body = re.sub(rf'\b{oracle_func}\b', pg_func, trigger_body)

        # Handle TO_DATE conversion
        trigger_body = re.sub(r'TO_DATE\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', r"TO_DATE(\1, 'YYYY-MM-DD')", trigger_body)

        # Convert DECODE to CASE statements
        def decode_to_case(match):
            args = match.group(1).split(',')
            case_stmt = "CASE " + args[0].strip()
            for i in range(1, len(args) - 1, 2):
                case_stmt += f" WHEN {args[i].strip()} THEN {args[i+1].strip()}"
            if len(args) % 2 == 0:
                case_stmt += f" ELSE {args[-1].strip()}"
            case_stmt += " END"
            return case_stmt

        trigger_body = re.sub(r'DECODE\s*\(([^)]+)\)', decode_to_case, trigger_body)

        # Convert sequences
        trigger_body = re.sub(r'(\w+)\.NEXTVAL', r"nextval('\1')", trigger_body)
        trigger_body = re.sub(r'(\w+)\.CURRVAL', r"currval('\1')", trigger_body)

        # Handle data type conversions
        trigger_body = trigger_body.replace('NUMBER', 'NUMERIC')
        trigger_body = trigger_body.replace('VARCHAR2', 'VARCHAR')

        # Convert CONNECT BY to WITH RECURSIVE (this is a simplified conversion and might need manual adjustment)
        if 'CONNECT BY' in trigger_body:
            print("Warning: CONNECT BY clause detected. This may require manual conversion to WITH RECURSIVE.")

        return trigger_body
    except Exception as e:
        print(f"Error in convert_trigger_body: {e}")
        return trigger_body

def migrate_trigger_to_postgres(pg_conn, schema_name, trigger_name, triggering_event, trigger_type, trigger_body, table_name):
    cursor = pg_conn.cursor()

    try:
        # Set the search path to the schema in PostgreSQL, preserving capitalization
        cursor.execute(f'SET search_path TO "{schema_name}";')

        # Check if the schema exists
        cursor.execute(f"SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = %s)", (schema_name,))
        schema_exists = cursor.fetchone()[0]

        if not schema_exists:
            print(f"Schema '{schema_name}' does not exist in PostgreSQL. Creating it now.")
            cursor.execute(f'CREATE SCHEMA "{schema_name}";')
            pg_conn.commit()

        pg_trigger_name = f"{trigger_name}"  # Keep original case

        # Check if the trigger already exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.triggers
                WHERE trigger_schema = %s
                AND event_object_table = %s
                AND trigger_name = %s
            )
        """, (schema_name, table_name, pg_trigger_name))

        trigger_exists = cursor.fetchone()[0]

        if trigger_exists:
            print(f"Trigger {pg_trigger_name} already exists in PostgreSQL. Skipping.")
            return

        pg_trigger_event = triggering_event.upper()
        pg_trigger_timing = "BEFORE" if "before" in trigger_type.lower() else "AFTER"

        # Translate Oracle trigger body to PostgreSQL PL/pgSQL
        pg_trigger_body = convert_trigger_body(trigger_body)

        # Create the new trigger directly on the table
        pg_trigger_sql = f"""
        CREATE OR REPLACE FUNCTION "{schema_name}"."{pg_trigger_name}_func"()
        RETURNS TRIGGER AS $$
        DECLARE
            -- Add any necessary variable declarations here
        BEGIN
            {pg_trigger_body.strip()}
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        CREATE TRIGGER "{pg_trigger_name}"
        {pg_trigger_timing} {pg_trigger_event} ON "{schema_name}"."{table_name}"
        FOR EACH ROW EXECUTE FUNCTION "{schema_name}"."{pg_trigger_name}_func"();
        """

        cursor.execute(pg_trigger_sql)
        pg_conn.commit()
        print(f"Trigger {trigger_name} migrated successfully.")

    except psycopg2.Error as e:
        pg_conn.rollback()  # Rollback transaction if there's an error
        print(f"Error executing trigger {trigger_name}: {e}")

    finally:
        cursor.close()

def migrate_triggers(oracle_dsn, oracle_user, oracle_password, pg_dsn, schema_name):
    oracle_conn = None
    pg_conn = None
    try:
        # Connect to Oracle and PostgreSQL
        oracle_conn = cx_Oracle.connect(oracle_user, oracle_password, oracle_dsn)
        pg_conn = psycopg2.connect(pg_dsn)

        # Ensure the schema exists in PostgreSQL
        cursor = pg_conn.cursor()
        cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}";')
        pg_conn.commit()
        cursor.close()

        triggers = get_oracle_triggers(oracle_conn, schema_name)

        for trigger in triggers:
            try:
                trigger_name, triggering_event, trigger_type, trigger_body, table_name = trigger
                migrate_trigger_to_postgres(pg_conn, schema_name, trigger_name, triggering_event, trigger_type, trigger_body, table_name)
            except Exception as e:
                print(f"Error migrating trigger {trigger[0]}: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()  # Print the traceback for more details

    finally:
        if oracle_conn:
            oracle_conn.close()
        if pg_conn:
            pg_conn.close()

if __name__ == "__main__":
    try:
        # Oracle connection details
        oracle_dsn = "promptoraserver:1521/xepdb1"
        oracle_user = "promptora"
        oracle_password = "1234"

        # PostgreSQL connection details
        pg_dsn = "dbname=postgres user=postgres password=promptora host=localhost"

        # Schema name to migrate
        schema_name = "HR"

        migrate_triggers(oracle_dsn, oracle_user, oracle_password, pg_dsn, schema_name)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)