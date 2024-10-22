import cx_Oracle
import psycopg2
import sys
import traceback
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def connect_oracle(username, password, host, port, service_name):
    logging.info(f"Connecting to Oracle: {host}:{port}/{service_name}")
    try:
        dsn = cx_Oracle.makedsn(host, port, service_name=service_name)
        return cx_Oracle.connect(username, password, dsn)
    except cx_Oracle.Error as error:
        logging.error(f"Error connecting to Oracle: {error}")
        raise

def connect_postgres(username, password, host, port, database):
    logging.info(f"Connecting to PostgreSQL: {host}:{port}/{database}")
    try:
        return psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=username,
            password=password
        )
    except psycopg2.Error as error:
        logging.error(f"Error connecting to PostgreSQL: {error}")
        raise

def get_oracle_packages(oracle_conn, schema_name):
    logging.info(f"Fetching packages from schema: {schema_name}")
    cursor = oracle_conn.cursor()
    try:
        cursor.execute(f"""
            SELECT object_name AS package_name, object_type
            FROM all_objects
            WHERE object_type IN ('PACKAGE', 'PACKAGE BODY')
            AND owner = '{schema_name.upper()}'
            ORDER BY object_name, object_type
        """)
        packages = cursor.fetchall()
        logging.info(f"Found {len(packages)} packages")
        return packages
    except cx_Oracle.Error as error:
        logging.error(f"Error fetching packages: {error}")
        raise
    finally:
        cursor.close()

def get_package_source(oracle_conn, schema_name, package_name, package_type):
    logging.info(f"Fetching source for {package_type}: {package_name}")
    cursor = oracle_conn.cursor()
    try:
        cursor.execute("""
            SELECT LISTAGG(TEXT, CHR(10)) WITHIN GROUP (ORDER BY LINE)
            FROM ALL_SOURCE
            WHERE OWNER = :schema_name
            AND NAME = :package_name
            AND TYPE = :package_type
        """, schema_name=schema_name, package_name=package_name, package_type=package_type)
        result = cursor.fetchone()
        if result is None or result[0] is None:
            logging.warning(f"No source found for {package_type}: {package_name}")
            return None
        source = result[0]
        logging.info(f"Source fetched, length: {len(source)} characters")
        return source
    except cx_Oracle.Error as error:
        logging.error(f"Error fetching package source: {error}")
        raise
    finally:
        cursor.close()

def remove_package_syntax(source):
    # Remove package and package body declarations
    source = re.sub(r'(CREATE\s+OR\s+REPLACE\s+)?PACKAGE(\s+BODY)?\s+\w+(\s+AS|\s+IS)', '', source, flags=re.IGNORECASE)
    # Remove 'END package_name;'
    source = re.sub(r'END\s+\w+;', 'END;', source, flags=re.IGNORECASE)
    return source

def convert_data_types(source):
    type_conversions = {
        'NUMBER': 'NUMERIC',
        'VARCHAR2': 'VARCHAR',
        'CLOB': 'TEXT',
        'DATE': 'TIMESTAMP',
        'SYS_REFCURSOR': 'REFCURSOR',
        'BOOLEAN': 'BOOLEAN',
        'LONG': 'TEXT',
        'NVARCHAR2': 'VARCHAR',
        'BINARY_INTEGER': 'INTEGER',
        'PLS_INTEGER': 'INTEGER',
    }
    for oracle_type, pg_type in type_conversions.items():
        source = re.sub(rf'\b{oracle_type}\b', pg_type, source, flags=re.IGNORECASE)
    return source

def convert_functions(source):
    function_conversions = {
        'NVL': 'COALESCE',
        'SYSDATE': 'CURRENT_TIMESTAMP',
        'TRUNC': 'DATE_TRUNC',
        'TO_DATE': 'TO_TIMESTAMP',
        'DECODE': 'CASE',
    }
    for oracle_func, pg_func in function_conversions.items():
        source = re.sub(rf'\b{oracle_func}\b', pg_func, source, flags=re.IGNORECASE)
    return source

def convert_procedural_syntax(source):
    source = re.sub(r':=', '=', source)  # Assignment operator
    source = re.sub(r'\bIS\b', 'AS', source, flags=re.IGNORECASE)  # 'IS' to 'AS'
    source = re.sub(r'\bEXCEPTION\b', 'EXCEPTION WHEN OTHERS THEN', source, flags=re.IGNORECASE)  # Basic exception handling
    source = re.sub(r'(\w+)%TYPE', 'TYPEOF(\1)', source)  # %TYPE to TYPEOF()
    source = re.sub(r'(\w+)%ROWTYPE', 'TYPEOF(\1)', source)  # %ROWTYPE to TYPEOF()
    return source

def convert_to_postgres(oracle_source):
    logging.info("Converting Oracle source to PostgreSQL")
    postgres_source = oracle_source
    postgres_source = remove_package_syntax(postgres_source)
    postgres_source = convert_data_types(postgres_source)
    postgres_source = convert_functions(postgres_source)
    postgres_source = convert_procedural_syntax(postgres_source)
    
    # Convert RETURN REFCURSOR to RETURNS REFCURSOR
    postgres_source = re.sub(r'\bRETURN\s+REFCURSOR\b', 'RETURNS REFCURSOR', postgres_source, flags=re.IGNORECASE)
    
    return postgres_source

def split_into_routines(source):
    # This regex pattern matches both FUNCTION and PROCEDURE declarations
    routine_pattern = re.compile(r'(FUNCTION|PROCEDURE)\s+(\w+)\s*(\((.*?)\))?\s*(RETURN\s+\w+)?\s*(AS|IS)\s*(\$\$|BEGIN)?(.*?)(END;?)(\$\$)?', re.DOTALL | re.IGNORECASE)
    
    routines = {}
    for match in routine_pattern.finditer(source):
        routine_type, routine_name, params, param_list, return_clause, as_is, begin_keyword, body, end_keyword, dollar_quotes = match.groups()
        
        # Clean up the parameter list
        params = params.strip('()') if params else ''
        
        # Ensure the function body is properly enclosed in BEGIN...END
        if not begin_keyword or begin_keyword.upper() != 'BEGIN':
            body = f"BEGIN\n{body}\nEND;"
        else:
            body = f"{begin_keyword}\n{body}\n{end_keyword}"
        
        # Construct the PostgreSQL function or procedure
        if routine_type.upper() == 'FUNCTION':
            pg_routine = f"""
CREATE OR REPLACE FUNCTION {{full_routine_name}}({params})
{return_clause if return_clause else 'RETURNS VOID'}
AS $$
{body}
$$ LANGUAGE plpgsql;
"""
        else:  # PROCEDURE
            pg_routine = f"""
CREATE OR REPLACE PROCEDURE {{full_routine_name}}({params})
AS $$
{body}
$$ LANGUAGE plpgsql;
"""
        routines[routine_name] = (routine_type.upper(), pg_routine.strip())
    
    return routines

def create_postgres_routines(postgres_conn, schema_name, package_name, package_source):
    logging.info(f"Creating functions and procedures in PostgreSQL for package: {schema_name}.{package_name}")
    with postgres_conn.cursor() as cursor:
        try:
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')
            
            # Create REFCURSOR type if it doesn't exist
            cursor.execute("DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'refcursor') THEN CREATE TYPE REFCURSOR; END IF; END $$;")
            
            routines = split_into_routines(package_source)
            
            for routine_name, (routine_type, routine_source) in routines.items():
                full_routine_name = f'"{schema_name}"."{package_name}_{routine_name}"'
                routine_source = routine_source.format(full_routine_name=full_routine_name)
                try:
                    cursor.execute(routine_source)
                    logging.info(f"Successfully created {routine_type}: {full_routine_name}")
                except psycopg2.Error as error:
                    logging.error(f"Error creating {routine_type} {full_routine_name}: {error}")
                    logging.error(f"Problematic SQL:\n{routine_source}")
                    postgres_conn.rollback()
                    # Continue with the next routine instead of raising an exception
                    continue
            
            postgres_conn.commit()
            logging.info(f"Routines for package {schema_name}.{package_name} created successfully")
        except psycopg2.Error as error:
            logging.error(f"Error creating PostgreSQL routines: {error}")
            postgres_conn.rollback()
            raise

def migrate_packages(oracle_conn, postgres_conn, schema_name):
    logging.info(f"Starting migration for schema: {schema_name}")
    packages = get_oracle_packages(oracle_conn, schema_name)
    for package_name, package_type in packages:
        try:
            logging.info(f"Migrating {package_type}: {package_name}")
            oracle_source = get_package_source(oracle_conn, schema_name, package_name, package_type)
            if oracle_source is None:
                logging.warning(f"Skipping {package_type}: {package_name} due to missing source")
                continue
            postgres_source = convert_to_postgres(oracle_source)
            create_postgres_routines(postgres_conn, schema_name, package_name, postgres_source)
            logging.info(f"Successfully migrated {package_type}: {package_name}")
        except Exception as e:
            logging.error(f"Error migrating {package_type}: {package_name}")
            logging.error(f"Error details: {str(e)}")
            traceback.print_exc()
    logging.info("Migration completed")

def main():
    schema_name = input("Enter the schema name to migrate: ")
    logging.info(f"Starting migration for schema: {schema_name}")

    # Oracle connection details
    oracle_username = "promptora"
    oracle_password = "1234"
    oracle_host = "promptoraserver"
    oracle_port = "1521"
    oracle_service_name = "xepdb1"

    # PostgreSQL connection details
    postgres_username = "postgres"
    postgres_password = "promptora"
    postgres_host = "localhost"
    postgres_port = "5432"
    postgres_database = "postgres"

    oracle_conn = None
    postgres_conn = None

    try:
        oracle_conn = connect_oracle(oracle_username, oracle_password, oracle_host, oracle_port, oracle_service_name)
        postgres_conn = connect_postgres(postgres_username, postgres_password, postgres_host, postgres_port, postgres_database)

        migrate_packages(oracle_conn, postgres_conn, schema_name)
    except Exception as e:
        logging.error(f"An error occurred during migration: {str(e)}")
        traceback.print_exc()
    finally:
        if oracle_conn:
            oracle_conn.close()
        if postgres_conn:
            postgres_conn.close()
        logging.info("Migration script finished")

if __name__ == "__main__":
    main()