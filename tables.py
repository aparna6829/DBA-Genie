import cx_Oracle
import psycopg2
from psycopg2 import sql
import json
import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import logging

# Configuration
log_filename = f"migration_log.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def set_search_path(conn, schema_name):
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name)))
    conn.commit()

def create_schema_and_user(schema_name,POSTGRES_CONNECTION_STRING):
    schema_name = schema_name.upper()
    with psycopg2.connect(POSTGRES_CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name)))
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (schema_name,))
            user_exists = cur.fetchone() is not None
            if not user_exists:
                cur.execute(sql.SQL("CREATE USER {} WITH PASSWORD %s").format(sql.Identifier(schema_name)), ('1234',))
            cur.execute(sql.SQL("GRANT USAGE, CREATE ON SCHEMA {} TO {}").format(sql.Identifier(schema_name), sql.Identifier(schema_name)))
            set_search_path(conn, schema_name)
            conn.commit()
    logging.info(f"Schema and user '{schema_name}' created successfully and search path set.")

def check_and_install_postgis(schema_name,POSTGRES_CONNECTION_STRING):
    with psycopg2.connect(POSTGRES_CONNECTION_STRING) as conn:
        set_search_path(conn, schema_name)
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'postgis'")
            if cur.fetchone() is None:
                cur.execute(f"CREATE EXTENSION postgis WITH SCHEMA \"{schema_name}\";")
                conn.commit()
                logging.info("PostGIS extension installed.")
            else:
                logging.info("PostGIS extension already installed.")

def extract_columns_to_config1(schema_name,ORACLE_CONNECTION_STRING):
    schema_name = schema_name.upper()
    with cx_Oracle.connect(ORACLE_CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT atc.table_name, atc.column_name, atc.data_type
                FROM all_tab_columns atc
                JOIN all_tables at ON atc.owner = at.owner AND atc.table_name = at.table_name
                WHERE atc.owner = :schema_name
                ORDER BY atc.table_name, atc.column_id  
            """, schema_name=schema_name)
            columns = cur.fetchall()

    config = {}
    for table, column, datatype in columns:
        if table not in config:
            config[table] = []
        config[table].append({
            "column": column,
            "datatype": datatype
        })

    with open(f"{schema_name}_columns_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return config

def analyze_datatypes(config):
    all_datatypes = set()
    datatype_counts = {}
    tables_with_special_types = {}
    common_datatypes = {"VARCHAR2", "NUMBER", "DATE", "CLOB", "BLOB", "TIMESTAMP(6)", "CHAR", "INTERVAL DAY(2) TO SECOND(6)", "SDO_KEYWORDARRAY", "SDO_GEOMETRY"}

    for table, columns in config.items():
        table_datatypes = set()
        for column in columns:
            datatype = column["datatype"]
            all_datatypes.add(datatype)
            table_datatypes.add(datatype)
            datatype_counts[datatype] = datatype_counts.get(datatype, 0) + 1

        special_types = table_datatypes - common_datatypes
        if special_types:
            tables_with_special_types[table] = list(special_types)

    return all_datatypes, datatype_counts, tables_with_special_types

def get_llm_suggestion(tables_with_special_types,schema_name):
    try:
        if not tables_with_special_types and schema_name =='HERE_SF':
            return "SDO_Geometry and SDO_Keywordarray found. Postgis seems to be installed. Standard migration to be sufficient"
        elif not tables_with_special_types:
            return "No special datatypes found. Standard migration should be sufficient."
        prompt = "I'm migrating an Oracle database to PostgreSQL. I have the following tables with special Oracle data types:\n\n"
        for table, datatypes in tables_with_special_types.items():
            prompt += f"Table '{table}': {', '.join(datatypes)}\n"
        prompt += "\nWhat PostgreSQL extensions or plugins would you recommend for handling these data types, and how should we approach migrating these tables?"
        GOOGLE_API_KEY = st.secrets["GEMINI_KEY"]
        model = ChatGoogleGenerativeAI(model_name="gemini-1.5-flash-latest",google_api_key=GOOGLE_API_KEY)
        response = model.invoke(prompt).content

        return response.text
    except:
        st.error("An error occured during LLM response generation")
def table_exists(conn, schema_name, table_name):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = %s AND table_name = %s
            )
        """, (schema_name, table_name))
        return cur.fetchone()[0]

def migrate_table_structure(schema_name, table_name, ORACLE_CONNECTION_STRING,POSTGRES_CONNECTION_STRING):
    schema_name = schema_name.upper()
    try:
        with cx_Oracle.connect(ORACLE_CONNECTION_STRING) as oracle_conn:
            with psycopg2.connect(POSTGRES_CONNECTION_STRING) as postgres_conn:
                set_search_path(postgres_conn, schema_name)
                
                # Check if the table already exists
                if table_exists(postgres_conn, schema_name, table_name):
                    logging.info(f"Table '{table_name}' already exists in schema '{schema_name}'. Skipping migration.")
                    return None

                oracle_cur = oracle_conn.cursor()
                postgres_cur = postgres_conn.cursor()

                oracle_cur.execute(f"""
                    SELECT column_name, data_type, data_length, data_precision, data_scale, nullable
                    FROM all_tab_columns
                    WHERE owner = :schema_name AND table_name = :table_name
                    ORDER BY column_id
                """, schema_name=schema_name, table_name=table_name)

                columns = oracle_cur.fetchall()

                create_table_sql = f'CREATE TABLE "{table_name}" ('
                column_defs = []
                columns_info = []

                for column in columns:
                    column_name, data_type, data_length, data_precision, data_scale, nullable = column
                    column_info = {
                        "column": column_name,
                        "datatype": data_type,
                        "nullable": nullable == 'Y'
                    }

                    if data_type == "VARCHAR2":
                        column_defs.append(f'"{column_name}" VARCHAR({data_length})')
                    elif data_type == "NUMBER":
                        if data_precision is not None and data_scale is not None:
                            column_defs.append(f'"{column_name}" NUMERIC({data_precision},{data_scale})')
                        else:
                            column_defs.append(f'"{column_name}" NUMERIC')
                    elif data_type == "DATE":
                        column_defs.append(f'"{column_name}" DATE')
                    elif data_type == "CLOB":
                        column_defs.append(f'"{column_name}" TEXT')
                    elif data_type == "BLOB":
                        column_defs.append(f'"{column_name}" BYTEA')
                    elif data_type == "CHAR":
                        column_defs.append(f'"{column_name}" CHAR({data_length})')
                    elif data_type == "INTERVAL DAY(2) TO SECOND(6)":
                        column_defs.append(f'"{column_name}" INTERVAL DAY TO SECOND(6)')
                    elif data_type == "TIMESTAMP(6)":
                        column_defs.append(f'"{column_name}" TIMESTAMP(6)')
                    elif data_type == "SDO_GEOMETRY":
                        geometry_type = get_geometry_type(oracle_conn, schema_name, table_name, column_name)
                        column_defs.append(f'"{column_name}" geometry({geometry_type}, 4326)')
                        column_info["geometry_type"] = geometry_type
                    elif data_type == "SDO_KEYWORDARRAY":
                        column_defs.append(f'"{column_name}" TEXT[]')
                    else:
                        column_defs.append(f'"{column_name}" {data_type}')

                    if nullable == 'N':
                        column_defs[-1] += " NOT NULL"

                    columns_info.append(column_info)

                create_table_sql += ", ".join(column_defs) + ");"
                postgres_cur.execute(create_table_sql)

                # Create spatial index if there's a geometry column
                # geometry_columns = [col["column"] for col in columns_info if col["datatype"] == "SDO_GEOMETRY"]
                # for geometry_column in geometry_columns:
                #     index_name = f"sidx_{table_name}_{geometry_column}"
                #     index_sql = f'CREATE INDEX {index_name} ON "{table_name}" USING GIST ("{geometry_column}");'
                #     postgres_cur.execute(index_sql)

                postgres_conn.commit()

        return columns_info

    except Exception as e:
        logging.info(f"Error migrating table structure {table_name}: {str(e)}")
        return None

def get_geometry_type(conn, schema_name, table_name, column_name):
    # Placeholder for geometry type extraction
    return "GEOMETRY"

def migrate_table_structures(schema_name, config, ORACLE_CONNECTION_STRING,POSTGRES_CONNECTION_STRING):
    with st.spinner("Migrating table schemas.. This might take a while: "):
        create_schema_and_user(schema_name,POSTGRES_CONNECTION_STRING)
        check_and_install_postgis(schema_name,POSTGRES_CONNECTION_STRING)

        all_datatypes, datatype_counts, tables_with_special_types = analyze_datatypes(config)
        llm_suggestion = get_llm_suggestion(tables_with_special_types,schema_name)
        st.write("LLM Suggestion:", llm_suggestion)

        tables_to_migrate = [table for table in config.keys() if table not in tables_with_special_types]
        migrated_tables = []

        for table in tables_to_migrate:
            logging.info(f"Migrating structure for table: {table}")
            result = migrate_table_structure(schema_name, table,ORACLE_CONNECTION_STRING,POSTGRES_CONNECTION_STRING)
            if result is not None:
                migrated_tables.append(table)

    
    logging.info(f"Migrated tables: {migrated_tables}")
    logging.info(f"Tables with special types (skipped): {tables_with_special_types}")
    return migrated_tables, tables_with_special_types




import cx_Oracle
import psycopg2
from psycopg2 import sql, Binary
import json
import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


def set_search_path(conn, schema_name):
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name)))
    conn.commit()

def adapt_oracle_type(value, datatype, oracle_conn):
    if value is None:
        return None
    if isinstance(value, cx_Oracle.LOB):
        if datatype == 'BLOB':
            return Binary(value.read())
        elif datatype == 'CLOB':
            return value.read()
    elif isinstance(value, cx_Oracle.Object):
        if datatype == 'SDO_KEYWORDARRAY':
            return '{' + ','.join(f'"{keyword}"' for keyword in value.aslist()) + '}'
        return str(value)
    elif isinstance(value, (list, tuple)):
        return '{' + ','.join(f'"{item}"' if isinstance(item, str) else str(item) for item in value) + '}'
    elif isinstance(value, str):
        # Handle potential array literals in string form
        if value.startswith('{') and value.endswith('}'):
            try:
                # Try to parse as JSON array
                array_items = json.loads(value.replace('{', '[').replace('}', ']'))
                return '{' + ','.join(f'"{item}"' if isinstance(item, str) else str(item) for item in array_items) + '}'
            except json.JSONDecodeError:
                # If it's not a valid JSON array, return as is
                return value
    return value

def check_table_exists(postgres_conn, table_name):
    with postgres_conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            )
        """, (table_name,))
        return cur.fetchone()[0]

def check_table_empty(postgres_conn, table_name):
    with postgres_conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name)))
        return cur.fetchone()[0] == 0

def migrate_table_data(schema_name, table_name, columns,ORACLE_CONNECTION_STRING,POSTGRES_CONNECTION_STRING,BATCH_SIZE,CHUNK_SIZE):
    try:
        with cx_Oracle.connect(ORACLE_CONNECTION_STRING) as oracle_conn, \
             psycopg2.connect(POSTGRES_CONNECTION_STRING) as postgres_conn:

            set_search_path(postgres_conn, schema_name)

            if not check_table_exists(postgres_conn, table_name):
                logging.info(f"Table {table_name} does not exist in PostgreSQL. Skipping.")
                return

            if not check_table_empty(postgres_conn, table_name):
                logging.info(f"Table {table_name} already contains data. Skipping.")
                return

            oracle_cur = oracle_conn.cursor()
            postgres_cur = postgres_conn.cursor()

            column_names = [col['column'] for col in columns]
            column_datatypes = {col['column']: col['datatype'] for col in columns}
            column_list = ', '.join(f'"{name}"' for name in column_names)

            oracle_cur.execute(f"""
                SELECT COUNT(*)
                FROM "{schema_name.upper()}"."{table_name}"
            """)
            total_rows = oracle_cur.fetchone()[0]

            offset = 0
            while offset < total_rows:
                oracle_cur.execute(f"""
                    SELECT {column_list}
                    FROM "{schema_name.upper()}"."{table_name}"
                    OFFSET {offset} ROWS FETCH NEXT {CHUNK_SIZE} ROWS ONLY
                """)
                rows = oracle_cur.fetchall()
                processed_rows = []
                for row in rows:
                    try:
                        processed_row = [adapt_oracle_type(value, column_datatypes[column_names[i]], oracle_conn) for i, value in enumerate(row)]
                        processed_rows.append(processed_row)
                    except Exception as e:
                        logging.info(f"Error processing row in table {table_name}: {str(e)}")
                        continue

                if processed_rows:
                    placeholders = []
                    for col in column_names:
                        if column_datatypes[col] == 'SDO_KEYWORDARRAY':
                            placeholders.append(sql.SQL('{}::text[]').format(sql.Placeholder()))
                        else:
                            placeholders.append(sql.Placeholder())
                    insert_sql = sql.SQL("""
                        INSERT INTO {} ({})
                        VALUES ({})
                    """).format(
                        sql.Identifier(table_name.upper()),
                        sql.SQL(', ').join(map(sql.Identifier, column_names)),
                        sql.SQL(', ').join(placeholders)
                    )
                    for i in range(0, len(processed_rows), BATCH_SIZE):
                        batch = processed_rows[i:i + BATCH_SIZE]
                        try:
                            postgres_cur.executemany(insert_sql, batch)
                            postgres_conn.commit()
                        except Exception as e:
                            # logging.info(f"Error inserting batch in table {table_name}: {str(e)}")
                            postgres_conn.rollback()
                offset += CHUNK_SIZE
                logging.info(f"Migrated rows {offset - CHUNK_SIZE} to {offset} for table {table_name}")
            logging.info(f"Data for table {table_name} migrated successfully.")

    except Exception as e:
        logging.info(f"Error migrating data for table {table_name}: {str(e)}")
        raise

def identify_sdo_geometry_tables(config):
    sdo_geometry_tables = []
    for table, columns in config.items():
        if any(col['datatype'] == 'SDO_GEOMETRY' for col in columns):
            sdo_geometry_tables.append(table)
    return sdo_geometry_tables

def save_sdo_geometry_tables(schema_name, sdo_geometry_tables):
    filename = f"{schema_name}_sdo_geometry_tables.json"
    with open(filename, "w") as f:
        json.dump({"schema": schema_name, "tables": sdo_geometry_tables}, f)
    logging.info(f"SDO_GEOMETRY tables saved to {filename}")

def table_has_data(postgres_conn, table_name):
    with postgres_conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT EXISTS (SELECT 1 FROM {} LIMIT 1)").format(sql.Identifier(table_name)))
        return cur.fetchone()[0]

def migrate_data(schema_name, config,ORACLE_CONNECTION_STRING,POSTGRES_CONNECTION_STRING,BATCH_SIZE=5000,CHUNK_SIZE=50000,MAX_WORKERS=16):
    start_time = time.time()

    logging.info(f"Starting data migration from schema '{schema_name}'...")

    sdo_geometry_tables = identify_sdo_geometry_tables(config)
    tables_to_migrate = [table for table in config if table not in sdo_geometry_tables]

    save_sdo_geometry_tables(schema_name, sdo_geometry_tables)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_table = {executor.submit(migrate_table_data, schema_name, table, config[table],ORACLE_CONNECTION_STRING,POSTGRES_CONNECTION_STRING,BATCH_SIZE,CHUNK_SIZE): table for table in tables_to_migrate}
        for future in as_completed(future_to_table):
            table = future_to_table[future]
            try:
                future.result()
            except Exception as exc:
                logging.info(f"{table} generated an exception: {exc}")

    end_time = time.time()






def get_system_info(oracle_connection_string):
    with cx_Oracle.connect(oracle_connection_string) as conn:
        cursor = conn.cursor()
        # Fetch SGA size
        cursor.execute("SELECT BYTES FROM v$sgainfo WHERE NAME = 'Maximum SGA Size'")
        sga_size = int(cursor.fetchone()[0])
        # Fetch CPU count
        cursor.execute("SELECT VALUE FROM v$osstat WHERE STAT_NAME = 'NUM_CPUS'")
        cpu_count = int(cursor.fetchone()[0])
        # Fetch available memory
        cursor.execute("SELECT VALUE FROM v$osstat WHERE STAT_NAME = 'FREE_MEMORY_BYTES'")
        available_memory = int(cursor.fetchone()[0])
        return sga_size, cpu_count, available_memory

def calculate_dynamic_values(sga_size, cpu_count, available_memory):
    # Calculate MAX_WORKERS based on CPU count
    max_workers = min(cpu_count * 8, 32)  # Cap at 32 workers

    

    # Calculate BATCH_SIZE based on SGA size
    batch_size = min(max(5000, sga_size // (1024 * 1024 * 10)), 50000)  # Between 5,000 and 50,000

    # Calculate CHUNK_SIZE based on available memory
    chunk_size = min(max(50000, available_memory // (1024 * 1024 * 100)), 500000)  # Between 50,000 and 500,000

    return max_workers, batch_size, chunk_size
    
    
    
    
import cx_Oracle
import psycopg2
from psycopg2 import sql, Binary
import json
import time
import os
import concurrent.futures
from tqdm import tqdm


def set_search_path1(conn, schema_name):
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name)))
    conn.commit()

def adapt_oracle_type(value, datatype, oracle_conn):
    if value is None:
        return None
    if isinstance(value, cx_Oracle.LOB):
        if datatype == 'BLOB':
            return Binary(value.read())
        elif datatype == 'CLOB':
            return value.read()
    elif isinstance(value, cx_Oracle.Object):
        if datatype == 'SDO_KEYWORDARRAY':
            return '{' + ','.join(f'"{keyword}"' for keyword in value.aslist()) + '}'
        return str(value)
    elif isinstance(value, (list, tuple)):
        return '{' + ','.join(map(str, value)) + '}'
    return value

def get_row_count(schema_name, table_name,ORACLE_CONNECTION_STRING):
    with cx_Oracle.connect(ORACLE_CONNECTION_STRING) as oracle_conn:
        with oracle_conn.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {schema_name}.{table_name}")
            return cursor.fetchone()[0]

def fetch_oracle_data(schema_name, table_name, columns, ORACLE_CONNECTION_STRING,batch_size,geom_column=None):
    with cx_Oracle.connect(ORACLE_CONNECTION_STRING) as oracle_conn:
        if geom_column:
            oracle_col_names = ', '.join([f'"{col["column"]}"' for col in columns if col["datatype"] != 'SDO_GEOMETRY'] + [f'SDO_UTIL.TO_WKTGEOMETRY({geom_column}) AS geom_wkt'])
        else:
            oracle_col_names = ', '.join([f'"{col["column"]}"' for col in columns])

        select_query = f"SELECT {oracle_col_names} FROM {schema_name}.{table_name}"

        with oracle_conn.cursor() as cursor:
            cursor.execute(select_query)
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                yield rows

def process_batch(batch):
    return [
        [value.read() if isinstance(value, cx_Oracle.LOB) else value for value in row]
        for row in batch
    ]

def migrate_all_table_data(schema_name, table_name, columns,ORACLE_CONNECTION_STRING,POSTGRES_CONNECTION_STRING,MAX_WORKERS,BATCH_SIZE):
    # Determine if the SDO_GEOMETRY column exists and its name
    geom_column = next((col['column'] for col in columns if col['datatype'] == 'SDO_GEOMETRY'), None)

    try:
        with psycopg2.connect(POSTGRES_CONNECTION_STRING) as postgres_conn:
            set_search_path1(postgres_conn, schema_name)

            if geom_column:
                column_names = [col['column'] for col in columns if col['datatype'] != 'SDO_GEOMETRY'] + [geom_column]
            else:
                column_names = [col['column'] for col in columns]

            col_placeholders = ', '.join(['%s' for _ in column_names])

            insert_query = sql.SQL("""
                INSERT INTO {} ({})
                VALUES ({})
            """).format(
                sql.Identifier(table_name.upper()),
                sql.SQL(', ').join(map(sql.Identifier, column_names)),
                sql.SQL(col_placeholders)
            )

            try:
                total_rows = get_row_count(schema_name, table_name,ORACLE_CONNECTION_STRING)
            except Exception as e:
                logging.info(f"Error getting row count for table {table_name}: {str(e)}")
                return False

            with tqdm(total=total_rows, desc=f"Migrating {table_name}") as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    with postgres_conn.cursor() as cursor:
                        try:
                            for batch in fetch_oracle_data(schema_name, table_name, columns,ORACLE_CONNECTION_STRING, BATCH_SIZE, geom_column):
                                processed_batch = list(executor.map(process_batch, [batch]))
                                cursor.executemany(insert_query, processed_batch[0])
                                pbar.update(len(batch))
                            postgres_conn.commit()
                        except Exception as e:
                            logging.info(f"Error processing batch for table {table_name}: {str(e)}")
                            postgres_conn.rollback()
                            return False

            st.write(f"Data for table {table_name} migrated successfully.")
            return True

    except Exception as e:
        logging.info(f"Error migrating data for table {table_name}: {str(e)}")
        return False

def identify_sdo_geometry_tables1(config):
    return [table for table, columns in config.items() if any(col['datatype'] == 'SDO_GEOMETRY' for col in columns)]

def migrate_complete_data(schema_name, config, migrated_tables,ORACLE_CONNECTION_STRING,POSTGRES_CONNECTION_STRING,MAX_WORKERS,BATCH_SIZE):
    start_time = time.time()

    st.write(f"Starting data migration from schema '{schema_name}'...")

    sdo_geometry_tables = identify_sdo_geometry_tables1(config)
    successful_migrations = []
    failed_migrations = []

    for table in migrated_tables:
        logging.info(f"Attempting to migrate data for table: {table}")
        table_columns = config.get(table, [])
        if migrate_all_table_data(schema_name, table, table_columns,ORACLE_CONNECTION_STRING,POSTGRES_CONNECTION_STRING,MAX_WORKERS,BATCH_SIZE):
            successful_migrations.append(table)
            logging.info(f"Successfully migrated data for the table: {table}")
        else:
            failed_migrations.append(table)

    end_time = time.time()
    st.write(f"Data migration completed in {end_time - start_time} seconds.")

    st.write(f"Tables with SDO_GEOMETRY: {sdo_geometry_tables}")
    st.write(f"Successfully migrated tables: {successful_migrations}")
    st.write(f"Failed migrations: {failed_migrations}")

    # Update the migrated_tables.json file
    with open("migrated_tables.json", "w") as f:
        json.dump(successful_migrations, f)

def load_or_create_migrated_tables(schema_name, config):
    filename = "migrated_tables.json"
    try:
        
            migrated_tables = list(config.keys())
            with open(filename, "w") as f:
                json.dump(migrated_tables, f)
            return migrated_tables
    except json.JSONDecodeError:
        logging.info(f"Error reading {filename}. Creating a new one with all tables.")
        migrated_tables = list(config.keys())
        with open(filename, "w") as f:
            json.dump(migrated_tables, f)
        return migrated_tables

    
    
    

def get_oracle_object_counts(schema_name,oracle_conn_str):
    oracle_conn = cx_Oracle.connect(oracle_conn_str)
    cursor = oracle_conn.cursor()

    # Define queries to get counts for different objects
    queries = {
        'tables': f"SELECT COUNT(*) FROM all_tables WHERE owner = '{schema_name}'",
        'views': f"SELECT COUNT(*) FROM all_views WHERE owner = '{schema_name}'",
        'indexes': f"SELECT COUNT(*) FROM all_indexes WHERE owner = '{schema_name}'",
        'functions': f"SELECT COUNT(*) FROM all_objects WHERE object_type = 'FUNCTION' AND owner = '{schema_name}'",
        'procedures': f"SELECT COUNT(*) FROM all_objects WHERE object_type = 'PROCEDURE' AND owner = '{schema_name}'",
        'triggers': f"SELECT COUNT(*) FROM all_triggers WHERE owner = '{schema_name}'",
        'sequences': f"SELECT COUNT(*) FROM all_sequences WHERE sequence_owner = '{schema_name}'",
        'dblinks' : f"SELECT COUNT(*) FROM all_db_links WHERE owner = '{schema_name}'"
        }

    counts = {}
    for obj_type, query in queries.items():
        cursor.execute(query)
        counts[obj_type] = cursor.fetchone()[0]

    cursor.close()
    return counts

def get_postgres_object_counts(schema_name,postgres_conn_str):
    postgres_conn = psycopg2.connect(postgres_conn_str)
    cursor = postgres_conn.cursor()

    # Define queries to get counts for different objects
    queries = {
        'tables': f"""SELECT COUNT(*) AS object_count FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = '{schema_name}' AND c.relkind = 'r' AND NOT EXISTS ( SELECT 1 FROM pg_extension e JOIN pg_depend d ON d.refobjid = e.oid WHERE d.objid = c.oid AND e.extname = 'postgis' ) GROUP BY c.relkind ORDER BY object_count DESC;""",
        'views': f"""SELECT COUNT(*) AS object_count FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = '{schema_name}' AND c.relkind = 'v' AND c.relname NOT LIKE 'geometry%' AND c.relname NOT LIKE 'geography%';""",
        'indexes': f"SELECT COUNT(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = '{schema_name}' AND c.relkind = 'i' AND c.relname != 'spatial_ref_sys_pkey';",
        'functions': f"""
SELECT COUNT(*) AS function_count
FROM (
    SELECT p.proname AS function_name
    FROM pg_proc p
    JOIN pg_namespace n ON n.oid = p.pronamespace
    LEFT JOIN pg_trigger t ON t.tgfoid = p.oid
    WHERE n.nspname = 'HR'
      AND p.prorettype != 'pg_catalog.trigger'::pg_catalog.regtype
      AND p.prokind != 'p'
      AND t.oid IS NULL
      AND p.proname NOT LIKE 'st_%'
      AND p.proname NOT LIKE 'SOUNDEX%'
      AND p.proname NOT LIKE 'elocation_%'
      AND p.proname NOT LIKE 'ELOCATION_%'
      AND p.proname NOT LIKE 'soundex%'
      AND p.proname NOT LIKE 'box%'
      AND p.proname NOT LIKE '_st_%'
      AND p.proname NOT LIKE 'bytea%'
      AND p.proname NOT LIKE 'dropgeo%'
      AND p.proname NOT LIKE 'checkauth%'
      AND p.proname NOT LIKE 'addgeom%'
      AND p.proname NOT LIKE 'addauth%'
      AND p.proname NOT LIKE 'contains_%'
      AND p.proname NOT LIKE 'equals%'
      AND p.proname NOT LIKE 'enablelong%'
      AND p.proname NOT LIKE 'disablelong%'
      AND p.proname NOT LIKE '_postgis%'
      AND p.proname NOT LIKE 'find_srid%'
      AND p.proname NOT LIKE 'geometry%'
      AND p.proname NOT LIKE 'geography%'
      AND p.proname NOT LIKE 'geog_brin%'
      AND p.proname NOT LIKE 'geom2d%'
      AND p.proname NOT LIKE 'geom3d%'
      AND p.proname NOT LIKE 'geom4d%'
      AND p.proname NOT LIKE 'geomfrom%'
      AND p.proname NOT LIKE 'get_proj4%'
      AND p.proname NOT LIKE 'gettransactionid%'
      AND p.proname NOT LIKE 'gidx_%'
      AND p.proname NOT LIKE 'gserialized_%'
      AND p.proname NOT LIKE 'is_contained%'
      AND p.proname NOT LIKE 'longtransactions%'
      AND p.proname NOT LIKE 'lockrow%'
      AND p.proname NOT LIKE 'overlaps_%'
      AND p.proname NOT LIKE 'json%'
      AND p.proname NOT LIKE 'path%'
      AND p.proname NOT LIKE 'pgis_%'
      AND p.proname NOT LIKE 'point%'
      AND p.proname NOT LIKE 'polygon%'
      AND p.proname NOT LIKE 'postgis_%'
      AND p.proname NOT LIKE 'populate_%'
      AND p.proname NOT LIKE 'spheroid%'
      AND p.proname NOT LIKE 'updategeo%'
      AND p.proname NOT LIKE 'text%'
      AND p.proname NOT LIKE 'unlockrows%'
) AS filtered_functions;
""",
        'procedures': f"select count(*) from pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace WHERE n.nspname = '{schema_name}' AND p.prokind = 'p';",
        'triggers' : f"SELECT COUNT(*) AS trigger_count FROM pg_trigger t JOIN pg_class c ON t.tgrelid = c.oid JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = '{schema_name}' AND NOT t.tgisinternal;",
        'sequences': f"SELECT COUNT(*) AS object_count FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = '{schema_name}' AND c.relkind = 'S';",
        'dblinks' : f"SELECT count(srvname) FROM pg_foreign_server fs JOIN pg_foreign_data_wrapper fdw ON fs.srvfdw = fdw.oid WHERE fdw.fdwname = 'postgres_fdw';"
        # 'synonyms': f"SELECT COUNT(*) FROM pg_views WHERE schemaname = '{schema_name}'"
    }

    counts = {}
    for obj_type, query in queries.items():
        cursor.execute(query)
        counts[obj_type] = cursor.fetchone()[0]

    cursor.close()
    return counts

    # Create a DataFrame to display the results
