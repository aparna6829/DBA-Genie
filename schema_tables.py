import cx_Oracle
import psycopg2
from psycopg2 import sql
import streamlit as st
import concurrent.futures
import time
import math
import json
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import cx_Oracle
from collections import defaultdict
import webbrowser
import cx_Oracle
import os
import json
from html import escape
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


import cx_Oracle
import psycopg2
from psycopg2 import sql
import streamlit as st
import concurrent.futures
import time
import math
import json
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import datetime
import cx_Oracle
import base64
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA,LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
import re
import pandas as pd
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_ollama import ChatOllama

GOOGLE_API_KEY = st.secrets["GEMINI_KEY"]

CHUNK_SIZE=50000
BATCH_SIZE=5000
MAX_WORKERS=15







import logging


log_filename = f"migration_log.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def create_schema_and_user(schema_name,postgres_conn_str):
    schema_name = schema_name.upper()  # Ensure schema name is in uppercase for PostgreSQL
    with psycopg2.connect(postgres_conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name)))
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (schema_name,))
            user_exists = cur.fetchone() is not None
            if not user_exists:
                cur.execute(sql.SQL("CREATE USER {} WITH PASSWORD %s").format(sql.Identifier(schema_name)), ('Promptora123',))
            cur.execute(sql.SQL("GRANT USAGE, CREATE ON SCHEMA {} TO {}").format(sql.Identifier(schema_name), sql.Identifier(schema_name)))
            conn.commit()
    st.write(f"Schema and user '{schema_name}' created successfully.")
 
def extract_columns_to_config(schema_name,oracle_conn_str):
    schema_name = schema_name.upper()  # Ensure schema name is in uppercase for Oracle
    with cx_Oracle.connect(oracle_conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT atc.table_name, atc.column_name, atc.data_type, atc.data_length, atc.data_precision, atc.data_scale
                FROM all_tab_columns atc
                JOIN all_tables at ON atc.owner = at.owner AND atc.table_name = at.table_name
                WHERE atc.owner = :schema_name
                ORDER BY atc.table_name, atc.column_id  
            """, schema_name=schema_name)
            columns = cur.fetchall()
 
    config = {}
    for table, column, datatype, length, precision, scale in columns:
        if table not in config:
            config[table] = []
        config[table].append({
            "column": column,
            "datatype": datatype,
            "length": length,
            "precision": precision,
            "scale": scale
        })
 
    with open(f"{schema_name}_columns_config.json", "w") as f:
        json.dump(config, f, indent=2)
 
    return config
 
def analyze_datatypes(config):
    all_datatypes = set()
    datatype_counts = {}
    tables_with_special_types = {}
    common_datatypes = {"VARCHAR2", "NUMBER", "DATE", "CLOB", "BLOB", "TIMESTAMP(6)", "CHAR", "INTERVAL DAY(2) TO SECOND(6)"}
 
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
 
def get_llm_suggestion(tables_with_special_types):
    if not tables_with_special_types:
        return "No special datatypes found. Standard migration should be sufficient."
    try:
    
        prompt = "I'm migrating an Oracle database to PostgreSQL. I have the following tables with special Oracle data types:\n\n"
        for table, datatypes in tables_with_special_types.items():
            prompt += f"Table '{table}': {', '.join(datatypes)}\n"
        prompt += "\nWhat PostgreSQL extensions or plugins would you recommend for handling these data types, and how should we approach migrating these tables?"
    
        GOOGLE_API_KEY = st.secrets["GEMINI_KEY"]
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
        response = llm.invoke(prompt)
    
        return response.content
    except:
        st.error("LLM access cannot be configured. Contact Promptora AI helpline:\nkeerthi@aipromptora.com\naparna@aipromptora.com\nganesh.gadhave@aipromptora.com")
 
def migrate_table_structure(schema_name, table_name, special_tables,oracle_conn_str,postgres_conn_str):
    schema_name = schema_name.upper()  # Ensure schema name is in uppercase for PostgreSQL
    if table_name in special_tables:
        st.write(f"Skipping table {table_name} due to special data types.")
        return None
 
    try:
        with cx_Oracle.connect(oracle_conn_str) as oracle_conn:
            with psycopg2.connect(postgres_conn_str) as postgres_conn:
                oracle_cur = oracle_conn.cursor()
                postgres_cur = postgres_conn.cursor()
 
                oracle_cur.execute(f"""
                    SELECT column_name, data_type, data_length, data_precision, data_scale, nullable
                    FROM all_tab_columns
                    WHERE owner = :schema_name AND table_name = :table_name
                    ORDER BY column_id
                """, schema_name=schema_name, table_name=table_name)
               
                columns = oracle_cur.fetchall()
               
                create_table_sql = f'CREATE TABLE "{schema_name}"."{table_name}" ('
                column_defs = []
 
                for column in columns:
                    column_name, data_type, data_length, data_precision, data_scale, nullable = column
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
                    else:
                        column_defs.append(f'"{column_name}" {data_type}')
                   
                    if nullable == 'N':
                        column_defs[-1] += " NOT NULL"
 
                create_table_sql += ", ".join(column_defs) + ");"
                postgres_cur.execute(create_table_sql)
                postgres_conn.commit()
                st.write(f"Table structure for {table_name} migrated successfully.")
       
        return [{"column": column_name, "datatype": data_type} for column_name, data_type, _, _, _, _ in columns]
   
    except Exception as e:
        st.error(f"Error migrating table structure {table_name}: {str(e)}")
        return None
 
def migrate_table_data(schema_name, table_name, columns,oracle_conn_str,postgres_conn_str):
    try:
        with cx_Oracle.connect(oracle_conn_str) as oracle_conn:
            with psycopg2.connect(postgres_conn_str) as postgres_conn:
                oracle_cur = oracle_conn.cursor()
                postgres_cur = postgres_conn.cursor()
 
                if not isinstance(columns, list) or not all(isinstance(col, dict) for col in columns):
                    raise ValueError(f"Invalid columns data format for table {table_name}")
 
                column_names = [col['column'] for col in columns]
                column_list = ', '.join(column_names)
 
                # Count total rows in the Oracle table
                oracle_cur.execute(f"""
                    SELECT COUNT(*)
                    FROM "{schema_name.upper()}"."{table_name}"
                """)
                total_rows = oracle_cur.fetchone()[0]
 
                st.write(f"Total rows to migrate for table {table_name}: {total_rows}")
 
                # Paginate through the rows
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
                        processed_row = []
                        for idx, value in enumerate(row):
                            if isinstance(value, cx_Oracle.LOB):
                                processed_row.append(value.read())  # Read LOB data
                            else:
                                processed_row.append(value)
                        processed_rows.append(processed_row)
 
                    if processed_rows:
                        insert_sql = sql.SQL("INSERT INTO {}.{} ({}) VALUES ({})").format(
                            sql.Identifier(schema_name),
                            sql.Identifier(table_name),
                            sql.SQL(', ').join(map(sql.Identifier, column_names)),
                            sql.SQL(', ').join(sql.Placeholder() * len(column_names))
                        )
 
                        # Batch size for inserts
                        for i in range(0, len(processed_rows), BATCH_SIZE):
                            batch = processed_rows[i:i + BATCH_SIZE]
                            postgres_cur.executemany(insert_sql, batch)
                            postgres_conn.commit()
 
                    offset += CHUNK_SIZE
                    st.write(f"Migrated rows {offset - CHUNK_SIZE} to {offset} for table {table_name}")
 
                st.write(f"Data for table {table_name} migrated successfully.")
 
    except Exception as e:
        st.error(f"Error migrating data for table {table_name}: {str(e)}")
 
def migrate_table_range(schema_name, tables, special_tables,oracle_conn_str,postgres_conn_str):
    table_structures = {}
    for table in tables:
        columns = migrate_table_structure(schema_name, table, special_tables,oracle_conn_str,postgres_conn_str)
        if columns:
            table_structures[table] = columns
    return table_structures
 
def migrate_schema_structure(schema_name,oracle_conn_str,postgres_conn_str,key):
    schema_name = schema_name.upper()  # Ensure schema name is in uppercase for PostgreSQL
    start_time = time.time()
   
    st.write(f"Creating schema and user '{schema_name}'...")
    create_schema_and_user(schema_name,postgres_conn_str)
   
    st.write(f"Extracting columns and datatypes for schema '{schema_name}'...")
    config = extract_columns_to_config(schema_name,oracle_conn_str)
   
    
    all_datatypes, datatype_counts, tables_with_special_types = analyze_datatypes(config)
   
    total_tables = len(config)
    special_tables_count = len(tables_with_special_types)
    with st.expander(f"\nAnalyzing datatypes for schema '{schema_name}'..."):
        st.write(f"\nTotal tables: {total_tables}")
        st.write(f"Tables with special datatypes: {special_tables_count}")
        st.write("\nDatatype counts:")
        for datatype, count in datatype_counts.items():
            st.write(f"{datatype}: {count}")
   
    if tables_with_special_types:
        with st.expander("\nTables with special datatypes:"):
            for table, datatypes in tables_with_special_types.items():
                st.write(f"Table '{table}': {', '.join(datatypes)}")
       
        with st.expander("\nGetting LLM suggestions for special datatypes..."):
            llm_suggestion = get_llm_suggestion(tables_with_special_types)
            st.write("LLM Suggestion:")
            st.write(llm_suggestion)
        st.divider()
    else:
        st.write("\nNo special datatypes found. Standard migration should be sufficient.")
   
    proceed = st.selectbox("Do you want to proceed with the migration?", ('yes', 'no'),index=None,key=key)
    if proceed == 'yes':
       
   
        tables = list(config.keys())
        tables_per_worker = math.ceil(len(tables) / MAX_WORKERS)
        table_ranges = [tables[i:i + tables_per_worker] for i in range(0, len(tables), tables_per_worker)]
    
        st.write(f"Migrating table structures in {len(table_ranges)} batches...")
        all_table_structures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(migrate_table_range, schema_name, table_range, tables_with_special_types,oracle_conn_str,postgres_conn_str) for table_range in table_ranges]
            for future in concurrent.futures.as_completed(futures):
                try:
                    all_table_structures.update(future.result())
                except Exception as e:
                    st.error(f"An error occurred in a batch: {e}")
    
        end_time = time.time()
        duration = end_time - start_time
    
    
        with open(f"{schema_name}_table_structures.json", "w") as f:
            json.dump(all_table_structures, f, indent=2)
        with st.expander("\nMigration Summary:"):
                st.write(f"Total tables: {total_tables}")
                st.write(f"Tables with special data types (skipped): {special_tables_count}")
                st.write(f"Tables successfully migrated: {len(all_table_structures)}")
                st.write("\nMigrating data...")

       
           
        migrate_schema_data(schema_name, all_table_structures, tables_with_special_types,oracle_conn_str,postgres_conn_str)
        return True
        
    if proceed == 'no':
        st.write("Migration aborted.")
        return False

def migrate_schema_data(schema_name, table_structures, special_tables,oracle_conn_str,postgres_conn_str):
    st.write(f"\nMigrating data for {len(table_structures)} tables...")
    total_tables = len(table_structures)
    progress_bar = st.progress(0)
    progress_text = st.empty()  # Create an empty placeholder for progress text
    elapsed_time_text = st.empty()  # Create an empty placeholder for elapsed time
    start_time = time.time()  # Record the start time
 
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(migrate_table_data, schema_name, table, columns,oracle_conn_str,postgres_conn_str): table for table, columns in table_structures.items() if table not in special_tables}
        completed_tables = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                st.error(f"An error occurred during data migration: {e}")
            completed_tables += 1
            progress_percentage = completed_tables / total_tables
            progress_bar.progress(progress_percentage)
            progress_text.text(f"Tables migrated: {completed_tables}/{total_tables}")
 
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            elapsed_time_text.text(f"Elapsed time: {elapsed_time:.2f} seconds")
 
    st.write("Data migration completed.")


def get_oracle_users(cursor):
    try:
        cursor.execute("""
            select distinct owner from dba_objects where owner not in ('ANONYMOUS','DBSFWUSER','PUBLIC','REMOTE_SCHEDULER_AGENT','APEX_040200','APEX_PUBLIC_USER','APPQOSSYS','AUDSYS','CTXSYS','DBSNMP','DVF','DVSYS','DIP','FLOWS_FILES','GSMADMIN_INTERNAL','GSMCATUSER','GSMUSER','LBACSYS','MDDATA','MDSYS','OJVMSYS','OLAPSYS','ORACLE_OCM','ORDPLUGINS','ORDDATA','ORDSYS','OUTLN','SI_INFORMTN_SCHEMA','SPATIAL_CSW_ADMIN_USR','SPATIAL_WFS_ADMIN_USR','SYS','SYSBACKUP','SYSKM','SYSDG','SYSTEM','WMSYS','XDB','XS$','XS$NULL') order by owner""")
        return [row[0].upper() for row in cursor.fetchall()]
    except:
        cursor.execute("""
            select distinct owner from all_objects where owner not in ('ANONYMOUS','DBSFWUSER','PUBLIC','REMOTE_SCHEDULER_AGENT','APEX_040200','APEX_PUBLIC_USER','APPQOSSYS','AUDSYS','CTXSYS','DBSNMP','DVF','DVSYS','DIP','FLOWS_FILES','GSMADMIN_INTERNAL','GSMCATUSER','GSMUSER','LBACSYS','MDDATA','MDSYS','OJVMSYS','OLAPSYS','ORACLE_OCM','ORDPLUGINS','ORDDATA','ORDSYS','OUTLN','SI_INFORMTN_SCHEMA','SPATIAL_CSW_ADMIN_USR','SPATIAL_WFS_ADMIN_USR','SYS','SYSBACKUP','SYSKM','SYSDG','SYSTEM','WMSYS','XDB','XS$','XS$NULL') order by owner""")
        return [row[0].upper() for row in cursor.fetchall()]
    
    
def get_the_Schema_name_from_user(chat_input):

    try:
        prompt = "You are a helpful assistant. When a user asks you to migrate their existing schema from OracleDB to PostgreSQL, you have to collect only the single user or multiple user names separated by a comma. Finally, when done, extract just only the single user name or multiple user names separated by a comma sentence and return output. Strictly instructed don't add other sentences before the user names.\nUser Query: " +chat_input
        GOOGLE_API_KEY = st.secrets["GEMINI_KEY"]
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
        response = llm.invoke(prompt)
        return response
    except:
        st.error("LLM access cannot be configured. Contact Promptora AI helpline:\nkeerthi@aipromptora.com\naparna@aipromptora.com\nganesh.gadhave@aipromptora.com")

def get_object_query():
    return """
    SELECT OBJECT_TYPE, COUNT(*) as OBJECT_COUNT
    FROM ALL_OBJECTS
    WHERE OWNER = :owner
    GROUP BY OBJECT_TYPE
    ORDER BY OBJECT_TYPE
    """

def generate_html_report(schema_info, object_data):
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .schema-info {{ margin-bottom: 20px; }}
            .object-type {{ font-weight: bold; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <h1>Oracle to Postgres - Database Migration Report</h1>
        <div class="schema-info">
            <p><strong>Version:</strong> {schema_info['version']}</p>
            <p><strong>Schema:</strong> {schema_info['schema']}</p>
            <p><strong>Size:</strong> {schema_info['size']}</p>
        </div>
        <table>
            <tr>
                <th>Object</th>
                <th>Number</th>
                <th>Invalid</th>
                <th>Estimated cost</th>
                <th>Comments</th>
                <th>Migration Solutions</th>
            </tr>
    """
    
    for obj_type, data in object_data.items():
        html += f"""
            <tr>
                <td>{obj_type}</td>
                <td>{data['count']}</td>
                <td>{data['invalid']}</td>
                <td>{data['estimated_cost']}</td>
                <td>{data['comments']}</td>
                <td>{data['details']}</td>
            </tr>
        """
    
    html += """
        </table>
    </body>
    </html>
    """
    return html

def get_schema_info(cursor,schema_name):
    cursor.execute("SELECT * FROM V$VERSION")
    version = cursor.fetchone()[0]
    cursor.execute("SELECT SUM(BYTES)/1024/1024 AS SIZE_MB FROM DBA_SEGMENTS WHERE OWNER = :schema_name", schema_name=schema_name)
    size = f"{cursor.fetchone()[0]:.2f}MB"
    return {'version': version, 'schema': schema_name, 'size': size}

def get_object_details(cursor, owner, object_type_query, object_type):
    cursor.execute(object_type_query, owner=owner)
    objects = cursor.fetchall()
    for obj_type, count in objects:
        if obj_type == object_type:
            return {
                'count': count,
                'invalid': 0,  # We'll estimate this based on typical rates
                'estimated_cost': estimate_cost(object_type, count),
                'comments': "",
                'details': ""
            }
    return None

def estimate_cost(object_type, count):
    # Estimated hours per object
    base_costs = {
        'DATABASE LINK': 0.5, 'GLOBAL TEMPORARY TABLE': 1, 'INDEX': 0.25, 'JOB': 2,
        'PROCEDURE': 3, 'SEQUENCE': 0.1, 'SYNONYM': 0.1, 'TABLE': 2, 'TRIGGER': 1.5, 'VIEW': 1
    }
    return round(base_costs.get(object_type, 1) * count, 2)

def get_all_object_details(object_type, object_name, temp_conn):
    cursor = temp_conn.cursor()
    query = ""
    
    if object_type.lower() == 'table':
        query = f"""
        SELECT column_name, data_type, data_length, nullable
        FROM all_tab_columns
        WHERE table_name = '{object_name.upper()}'
        """
    elif object_type.lower() == 'view':
        query = f"""
        SELECT text
        FROM all_views
        WHERE view_name = '{object_name.upper()}'
        """
    elif object_type.lower() == 'index':
        query = f"""
        SELECT index_type, uniqueness, tablespace_name
        FROM all_indexes
        WHERE index_name = '{object_name.upper()}'
        """
    elif object_type.lower() == 'sequence':
        query = f"""
        SELECT min_value, max_value, increment_by, cycle_flag
        FROM all_sequences
        WHERE sequence_name = '{object_name.upper()}'
        """
    elif object_type.lower() == 'procedure' or object_type.lower() == 'function':
        query = f"""
        SELECT text
        FROM all_source
        WHERE name = '{object_name.upper()}' AND type = '{object_type.upper()}'
        ORDER BY line
        """
    elif object_type.lower() == 'trigger':
        query = f"""
        SELECT trigger_type, triggering_event, table_name, trigger_body
        FROM all_triggers
        WHERE trigger_name = '{object_name.upper()}'
        """
    else:
        return "Unsupported object type"

    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    return str(results)

def setup(user_inputs):
    # Initialize the language models
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=st.secrets["GEMINI_KEY"])
        input_parser_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=st.secrets["GEMINI_KEY"])
        # Set up the Oracle database connection
        connection_string = f"oracle+cx_oracle://{user_inputs['oracle_user']}:{user_inputs['oracle_password']}@{user_inputs['oracle_host']}:{user_inputs['oracle_port']}/?service_name={user_inputs['oracle_service_name']}"
        temp_conn = cx_Oracle.connect(f"{user_inputs['oracle_user']}/{user_inputs['oracle_password']}@//{user_inputs['oracle_host']}:{user_inputs['oracle_port']}/{user_inputs['oracle_service_name']}")

        # Create SQLDatabase instance
        db = SQLDatabase.from_uri(connection_string)

        # Create SQLDatabaseToolkit
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        input_parser_prompt = PromptTemplate(
            input_variables=["user_input"],
            template="""
            Extract the following information from the user input:
            1. Object Type
            2. Object Name
            3. User Query (if any)

            User Input: {user_input}

            Strictly provide the extracted information in the following format:
            Object Type: [extracted object type]
            Object Name: [extracted object name]
            User Query: [extracted user query or 'None' if not present]
            """
        )
        input_parser_chain = LLMChain(llm=input_parser_llm, prompt=input_parser_prompt)
        input_parser_tool = Tool(
            name="Input Parser",
            func=input_parser_chain.run,
            description="Useful for parsing user input to extract object type, object name, and user query."
        )

        # Define the SQL query template
        sql_query_template = """
        You are given the following Oracle database schema object details, which are intended for migration to PostgreSQL:

        Object Type: {object_type}
        Object Name: {object_name}
        Object Details: {object_data}

        Based on the provided details, perform the following tasks:

        1. Provide a summary that specifically highlights the key aspects of the Oracle object `{object_name}`, including its structure, relationships, and any unique attributes.
        2. Outline detailed, step-by-step migration instructions tailored to the specific features and data types of the `{object_type}` `{object_name}`, including any special considerations for this particular object.
        3. Identify potential challenges specific to the data and structure of `{object_name}` when migrating to PostgreSQL, such as data type mismatches, index compatibility, constraint differences, or performance issues.
        4. Propose solutions for each identified challenge, ensuring that they address the unique characteristics of the `{object_name}`.
        5. Specify any PostgreSQL extensions that are necessary based on the features and requirements of `{object_name}`, and provide detailed installation and configuration steps.

        Additional user query: {user_query}

        Your output should be a concise summary in plain text format, addressing both the migration aspects and the user's specific query.
        """

        # Initialize the main agent
        agent = initialize_agent(
            toolkit.get_tools(),
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
        )

        # Initialize the input parser agent
        input_parser_agent = initialize_agent(
            [input_parser_tool],
            input_parser_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
        )
        return sql_query_template, agent, input_parser_agent, temp_conn
    except:
        st.error("LLM access cannot be configured. Contact Promptora AI helpline:\nkeerthi@aipromptora.com\naparna@aipromptora.com\nganesh.gadhave@aipromptora.com")

def generate_migration_summary(object_type, object_name, user_query, sql_query_template, agent, temp_conn):
    object_data = get_all_object_details(object_type, object_name, temp_conn)
    
    sql_query = sql_query_template.format(
        object_type=object_type,
        object_name=object_name,
        object_data=object_data,
        user_query=user_query
    )
    
    response = agent.run(sql_query)
    return response

def parse_user_input(user_input, input_parser_agent):
    response = input_parser_agent.run(f"Parse the following user input: {user_input}")
    return response

def generate_llm_insights(object_data):
    try:
        GOOGLE_API_KEY = st.secrets["GEMINI_KEY"]
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
            
        prompt_template = PromptTemplate(input_variables=["input"], template="{input}")
        llm_chain = LLMChain(prompt=prompt_template, llm=llm)
        
        for obj_type, data in object_data.items():
            prompt = f"""
            Provide insights for migrating {data['count']} {obj_type}(s) from Oracle to PostgreSQL:
            1. Brief comment (max 200 characters) on potential challenges.
            2. Detailed solutions (max 500 characters) for addressing these challenges during migration.
            Format the response as JSON with keys 'comment' and 'solutions'.
            """
            response = llm_chain.invoke({"input": prompt})
            try:
                res8=response['text'].split('```json')[1].split('```')[0]
                insights = json.loads(res8)
                object_data[obj_type]['comments'] = insights.get('comment', '')
                object_data[obj_type]['details'] = insights.get('solutions', '')
            except json.JSONDecodeError:
                insights = json.loads(response['text'])
                object_data[obj_type]['comments'] = insights.get('comment', '')
                object_data[obj_type]['details'] = insights.get('solutions', '')
        return object_data
    except:
        st.error("LLM access cannot be configured. Contact Promptora AI helpline:\nkeerthi@aipromptora.com\naparna@aipromptora.com\nganesh.gadhave@aipromptora.com")

def save_detailed_results(object_data, filename):
    with open(filename, 'w') as f:
        json.dump(object_data, f, indent=2)
        
def open_file(file):
    webbrowser.open(file)
    
def get_oracle_table_details(schema_name, oracle_conn_str):
    # Connect to Oracle
    connection = cx_Oracle.connect(oracle_conn_str)
    cursor = connection.cursor()

    # Query to get table details
    query = f"""
    SELECT table_name, column_name, data_type, data_length
    FROM all_tab_columns
    WHERE owner = '{schema_name.upper()}'
    ORDER BY table_name, column_id
    """

    cursor.execute(query)
    oracle_tables = cursor.fetchall()

    # Close the connection
    cursor.close()
    connection.close()

    return oracle_tables

def get_postgres_table_details(schema_name, user, password, host, port, database):
    # Connect to PostgreSQL
    connection = psycopg2.connect(
        dbname=database,
        user=user,
        password=password,
        host=host,
        port=port
    )
    cursor = connection.cursor()

    # Query to get table details
    query = f"""
    SELECT table_name, column_name, data_type, character_maximum_length
    FROM information_schema.columns
    WHERE table_schema = '{schema_name.lower()}'
    ORDER BY table_name, ordinal_position
    """

    cursor.execute(query)
    postgres_tables = cursor.fetchall()
    # print(postgres_tables)

    # Close the connection
    cursor.close()
    connection.close()

    return postgres_tables

def llm_validation(get_oracle_table_details,get_postgres_table_details):
    try:
        prompt = f"You need to compare the oracle_schema and postgres_schema and generate a concise report:\n\n{get_oracle_table_details}{get_postgres_table_details}"
        GOOGLE_API_KEY = st.secrets["GEMINI_KEY"]
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
        response = llm.invoke(prompt)

        # Display response from language model
        with st.expander("Response from language model:"):
            st.write(f"```{response.content.strip()}")
    except:
        st.error("LLM access cannot be configured. Contact Promptora AI helpline:\nkeerthi@aipromptora.com\naparna@aipromptora.com\nganesh.gadhave@aipromptora.com")
