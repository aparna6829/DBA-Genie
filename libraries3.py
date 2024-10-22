import webbrowser
import datetime
import cx_Oracle
import psycopg2
from psycopg2 import sql
import streamlit as st
import os
from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import time
import json
import base64
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA,LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
import re
import streamlit as st
import pandas as pd
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
load_dotenv()
 
file_path = "config_file.txt"
file_json = "config_file.json"

import logging


log_filename = f"migration_log.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def extract_db_details(prompt):
    GOOGLE_API_KEY = st.secrets["GEMINI_KEY"]
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
    except:
        st.error("LLM access cannot be configured. Contact Promptora AI helpline:\nkeerthi@aipromptora.com\naparna@aipromptora.com\nganesh.gadhave@aipromptora.com")
   
    system_message = """
    Extract the following information from the user's input:
    - Oracle Username
    - Oracle Host
    - Oracle Port
    - Oracle Service Name
    
    Format the output exactly like this, replacing the values in quotes:
    {
        "Oracle Username": "extracted_username",
        "Oracle Host": "extracted_host",
        "Oracle Port": "extracted_port",
        "Oracle Service Name": "extracted_service_name"
    }
    """

    # Create the user message with the prompt
    user_message = f"Extract the database connection details from this prompt: {prompt}"

    # Combine system and user messages
    full_prompt = system_message + "\n\n" + user_message

    # Generate the response from the LLM
    try:
        response = llm.predict(text=full_prompt)
        print("response",response)  # Use 'text' as the keyword argument
    except Exception as e:
        print(f"Error during LLM prediction: {e}")
        return None

    # Use regex to extract key-value pairs from the response
    pattern = r'"([^"]+)":\s*"([^"]+)"'
    matches = re.findall(pattern, response)

    # Check if matches were found
    if matches:
        # Construct a dictionary from the matches
        connection_details = {key.strip(): value.strip() for key, value in matches}
        
        # Ensure all required fields are present
        required_fields = ["Oracle Username", "Oracle Host", "Oracle Port", "Oracle Service Name"]
        if all(field in connection_details for field in required_fields):
            return connection_details
        else:
            print("Incomplete details extracted from LLM response")
            print("Extracted Details:", connection_details)
            return None
    else:
        print("Failed to extract connection details from LLM response")
        print("LLM Response:", response)
        return None
def extract_postgres_db_details(prompt):
    GOOGLE_API_KEY = st.secrets["GEMINI_KEY"]
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
    except:
        st.error("LLM access cannot be configured. Contact Promptora AI helpline:\nkeerthi@aipromptora.com\naparna@aipromptora.com\nganesh.gadhave@aipromptora.com")
   
    system_message = """
    Extract the following information from the user's input:
    - user
    - host
    - port
    - dbname
    
    Format the output exactly like this, replacing the values in quotes:
    {
        "username": "extracted_username",
        "host": "extracted_host",
        "port": "extracted_port",
        "dbname": "extracted_dbname"
    }
    """

    # Create the user message with the prompt
    user_message = f"Extract the database connection details from this prompt: {prompt}"

    # Combine system and user messages
    full_prompt = system_message + "\n\n" + user_message

    # Generate the response from the LLM
    try:
        response = llm.predict(text=full_prompt)
        print("response",response)  # Use 'text' as the keyword argument
    except Exception as e:
        print(f"Error during LLM prediction: {e}")
        return None

    # Use regex to extract key-value pairs from the response
    pattern = r'"([^"]+)":\s*"([^"]+)"'
    matches = re.findall(pattern, response)

    # Check if matches were found
    if matches:
        # Construct a dictionary from the matches
        connection_details = {key.strip(): value.strip() for key, value in matches}
        
        # Ensure all required fields are present
        required_fields = ["username", "host", "port", "dbname"]
        if all(field in connection_details for field in required_fields):
            return connection_details
        else:
            print("Incomplete details extracted from LLM response")
            print("Extracted Details:", connection_details)
            return None
    else:
        print("Failed to extract connection details from LLM response")
        print("LLM Response:", response)
        return None
def default_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, cx_Oracle.LOB):
        try:
            return obj.read().decode('utf-8')
        except AttributeError:
            return str(obj)
    elif isinstance(obj, (bytes, bytearray)):
        return base64.b64encode(obj).decode('utf-8')
    raise TypeError(f"Type {obj.__class__.__name__} not serializable")
 
def save_config(file_path, config_data):
    try:
        with open(file_path, 'w') as config_file:
            json.dump(config_data, config_file, indent=4, default=default_serializer)
    except IOError as e:
        raise IOError(f"Error saving configuration file: {e}")




def get_all_objects(cursor):
    query = """
    SELECT OWNER, OBJECT_TYPE, OBJECT_NAME
    FROM DBA_OBJECTS
    ORDER BY OWNER, OBJECT_TYPE, OBJECT_NAME
    """
    cursor.execute(query)
    return cursor.fetchall()

def get_all_details(cursor):
    queries = {
        'TABLE_VIEW': """
        SELECT OWNER, OBJECT_TYPE, OBJECT_NAME,
               COLUMN_NAME, DATA_TYPE, DATA_LENGTH, NULLABLE
        FROM (
            SELECT OWNER, 'TABLE' AS OBJECT_TYPE, TABLE_NAME AS OBJECT_NAME,
                   COLUMN_NAME, DATA_TYPE, DATA_LENGTH, NULLABLE
            FROM DBA_TAB_COLUMNS
            UNION ALL
            SELECT OWNER, 'VIEW' AS OBJECT_TYPE, TABLE_NAME AS OBJECT_NAME,
                   COLUMN_NAME, DATA_TYPE, DATA_LENGTH, NULLABLE
            FROM DBA_TAB_COLUMNS
            WHERE TABLE_NAME IN (SELECT VIEW_NAME FROM DBA_VIEWS)
        )
        """,
        'PROCEDURE': """
        SELECT OWNER, 'PROCEDURE' AS OBJECT_TYPE, OBJECT_NAME,
               CAST(NULL AS VARCHAR2(128)) AS COLUMN_NAME,
               DATA_TYPE,
               CAST(NULL AS NUMBER) AS DATA_LENGTH,
               CAST(NULL AS VARCHAR2(1)) AS NULLABLE,
               ARGUMENT_NAME, IN_OUT
        FROM DBA_ARGUMENTS
        """,
        'INDEX': """
        SELECT OWNER, 'INDEX' AS OBJECT_TYPE, INDEX_NAME AS OBJECT_NAME,
               CAST(NULL AS VARCHAR2(128)) AS COLUMN_NAME,
               CAST(NULL AS VARCHAR2(128)) AS DATA_TYPE,
               CAST(NULL AS NUMBER) AS DATA_LENGTH,
               CAST(NULL AS VARCHAR2(1)) AS NULLABLE,
               CAST(NULL AS VARCHAR2(128)) AS ARGUMENT_NAME,
               CAST(NULL AS VARCHAR2(9)) AS IN_OUT,
               INDEX_TYPE, UNIQUENESS, TABLESPACE_NAME
        FROM DBA_INDEXES
        """,
        'TRIGGER': """
        SELECT OWNER, 'TRIGGER' AS OBJECT_TYPE, TRIGGER_NAME AS OBJECT_NAME,
               CAST(NULL AS VARCHAR2(128)) AS COLUMN_NAME,
               CAST(NULL AS VARCHAR2(128)) AS DATA_TYPE,
               CAST(NULL AS NUMBER) AS DATA_LENGTH,
               CAST(NULL AS VARCHAR2(1)) AS NULLABLE,
               CAST(NULL AS VARCHAR2(128)) AS ARGUMENT_NAME,
               CAST(NULL AS VARCHAR2(9)) AS IN_OUT,
               CAST(NULL AS VARCHAR2(27)) AS INDEX_TYPE,
               CAST(NULL AS VARCHAR2(9)) AS UNIQUENESS,
               CAST(NULL AS VARCHAR2(30)) AS TABLESPACE_NAME,
               TRIGGER_TYPE, TRIGGERING_EVENT, TABLE_NAME, STATUS
        FROM DBA_TRIGGERS
        """,
        'SEQUENCE': """
        SELECT SEQUENCE_OWNER AS OWNER, 'SEQUENCE' AS OBJECT_TYPE, SEQUENCE_NAME AS OBJECT_NAME,
               CAST(NULL AS VARCHAR2(128)) AS COLUMN_NAME,
               CAST(NULL AS VARCHAR2(128)) AS DATA_TYPE,
               CAST(NULL AS NUMBER) AS DATA_LENGTH,
               CAST(NULL AS VARCHAR2(1)) AS NULLABLE,
               CAST(NULL AS VARCHAR2(128)) AS ARGUMENT_NAME,
               CAST(NULL AS VARCHAR2(9)) AS IN_OUT,
               CAST(NULL AS VARCHAR2(27)) AS INDEX_TYPE,
               CAST(NULL AS VARCHAR2(9)) AS UNIQUENESS,
               CAST(NULL AS VARCHAR2(30)) AS TABLESPACE_NAME,
               CAST(NULL AS VARCHAR2(16)) AS TRIGGER_TYPE,
               CAST(NULL AS VARCHAR2(4000)) AS TRIGGERING_EVENT,
               CAST(NULL AS VARCHAR2(128)) AS TABLE_NAME,
               CAST(NULL AS VARCHAR2(8)) AS STATUS,
               MIN_VALUE, MAX_VALUE, INCREMENT_BY, CYCLE_FLAG, ORDER_FLAG, CACHE_SIZE, LAST_NUMBER
        FROM DBA_SEQUENCES
        """
    }

    all_details = []
    for query in queries.values():
        cursor.execute(query)
        all_details.extend(cursor.fetchall())

    return all_details

def get_all_dependencies(cursor):
    query = """
    SELECT OWNER, NAME, TYPE, 
           REFERENCED_OWNER, REFERENCED_NAME, REFERENCED_TYPE,
           DEPENDENCY_TYPE
    FROM DBA_DEPENDENCIES
    """
    cursor.execute(query)
    return cursor.fetchall()

def get_view_definitions(cursor):
    query = """
    SELECT OWNER, VIEW_NAME, TEXT
    FROM DBA_VIEWS
    """
    cursor.execute(query)
    return cursor.fetchall()


def get_all_information(cursor):
    all_objects = get_all_objects(cursor)
    all_details = get_all_details(cursor)
    all_dependencies = get_all_dependencies(cursor)
    view_definitions = get_view_definitions(cursor)

    all_info = {}
    for owner, object_type, object_name in all_objects:
        if owner not in all_info:
            all_info[owner] = {}
        if object_type not in all_info[owner]:
            all_info[owner][object_type] = {}
        
        all_info[owner][object_type][object_name] = {
            'details': [],
            'intra_dependencies': [],
            'inter_dependencies': []
        }

    for row in all_details:
        owner, object_type, object_name = row[:3]
        if owner in all_info and object_type in all_info[owner] and object_name in all_info[owner][object_type]:
            detail_dict = dict(zip([
                'COLUMN_NAME', 'DATA_TYPE', 'DATA_LENGTH', 'NULLABLE',
                'ARGUMENT_NAME', 'IN_OUT',
                'INDEX_TYPE', 'UNIQUENESS', 'TABLESPACE_NAME',
                'TRIGGER_TYPE', 'TRIGGERING_EVENT', 'TABLE_NAME', 'STATUS',
                'MIN_VALUE', 'MAX_VALUE', 'INCREMENT_BY', 'CYCLE_FLAG', 'ORDER_FLAG', 'CACHE_SIZE', 'LAST_NUMBER'
            ], row[3:]))
            all_info[owner][object_type][object_name]['details'].append(detail_dict)

    # Fetch trigger bodies separately
    trigger_body_query = """
SELECT OWNER, TRIGGER_NAME, TRIGGER_BODY
FROM DBA_TRIGGERS
"""
    cursor.execute(trigger_body_query)
    for owner, trigger_name, trigger_body in cursor:
        if owner in all_info and 'TRIGGER' in all_info[owner] and trigger_name in all_info[owner]['TRIGGER']:
            all_info[owner]['TRIGGER'][trigger_name]['trigger_body'] = trigger_body

    for row in all_dependencies:
        owner, name, type, ref_owner, ref_name, ref_type, dep_type = row
        if owner in all_info and type in all_info[owner] and name in all_info[owner][type]:
            dep_info = {
                'REFERENCED_OWNER': ref_owner,
                'REFERENCED_NAME': ref_name,
                'REFERENCED_TYPE': ref_type,
                'DEPENDENCY_TYPE': dep_type
            }
            if owner == ref_owner:
                all_info[owner][type][name]['intra_dependencies'].append(dep_info)
            else:
                all_info[owner][type][name]['inter_dependencies'].append(dep_info)

    for owner, view_name, text in view_definitions:
        if owner in all_info and 'VIEW' in all_info[owner] and view_name in all_info[owner]['VIEW']:
            all_info[owner]['VIEW'][view_name]['view_definition'] = text

    return all_info

def generate_table_html(data, headers):
    if not data:
        return "<p>No data available</p>"
    
    table_html = '<table border="1"><tr>'
    for header in headers:
        table_html += f'<th>{header}</th>'
    table_html += '</tr>'
    for row in data:
        table_html += '<tr>'
        for header in headers:
            table_html += f'<td>{row.get(header, "")}</td>'
        table_html += '</tr>'
    table_html += '</table>'
    return table_html

def generate_object_html(object_name, object_type, details):
    html_content = f'<li><span class="caret">{object_name} ({object_type})</span><ul class="nested">'

    if details["details"]:
        html_content += '<li><span class="caret">Details</span><ul class="nested">'
        if object_type in ['TABLE', 'VIEW']:
            html_content += generate_table_html(
                details["details"],
                ["COLUMN_NAME", "DATA_TYPE", "DATA_LENGTH", "NULLABLE"]
            )
        elif object_type == 'PROCEDURE':
            html_content += generate_table_html(
                details["details"],
                ["ARGUMENT_NAME", "DATA_TYPE", "IN_OUT"]
            )
        elif object_type == 'INDEX':
            html_content += generate_table_html(
                details["details"],
                ["INDEX_TYPE", "UNIQUENESS", "TABLESPACE_NAME"]
            )
        elif object_type == 'TRIGGER':
            html_content += generate_table_html(
                details["details"],
                ["TRIGGER_TYPE", "TRIGGERING_EVENT", "TABLE_NAME", "STATUS", "TRIGGER_BODY"]
            )
        elif object_type == 'SEQUENCE':
            html_content += generate_table_html(
                details["details"],
                ["MIN_VALUE", "MAX_VALUE", "INCREMENT_BY", "CYCLE_FLAG", "ORDER_FLAG", "CACHE_SIZE", "LAST_NUMBER"]
            )
        html_content += "</ul></li>"

    if object_type == 'VIEW' and 'view_definition' in details:
        html_content += '<li><span class="caret">View Definition</span><ul class="nested">'
        html_content += f'<pre>{details["view_definition"]}</pre>'
        html_content += "</ul></li>"

    if details["intra_dependencies"]:
        html_content += '<li><span class="caret">Intra-Dependencies</span><ul class="nested">'
        html_content += generate_table_html(
            details["intra_dependencies"],
            ["REFERENCED_OWNER", "REFERENCED_NAME", "REFERENCED_TYPE", "DEPENDENCY_TYPE"]
        )
        html_content += "</ul></li>"

    if details["inter_dependencies"]:
        html_content += '<li><span class="caret">Inter-Dependencies</span><ul class="nested">'
        html_content += generate_table_html(
            details["inter_dependencies"],
            ["REFERENCED_OWNER", "REFERENCED_NAME", "REFERENCED_TYPE", "DEPENDENCY_TYPE"]
        )
        html_content += "</ul></li>"

    html_content += "</ul></li>"
    return html_content

def generate_html(organized_data):
    html_content = """
    <html>
    <head>
        <style>
            ul { list-style-type: none; }
            .caret { cursor: pointer; user-select: none; }
            .caret::before { content: "\\25B6"; color: black; display: inline-block; margin-right: 6px; }
            .caret-down::before { transform: rotate(90deg); }
            .nested { display: none; }
            .active { display: block; }
            table { border-collapse: collapse; margin-top: 10px; }
            th, td { border: 1px solid black; padding: 5px; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Oracle Database Information</h1>
        <ul id="myUL">
    """

    for owner, types in organized_data.items():
        html_content += f'<li><span class="caret">{owner}</span><ul class="nested">'
        for object_type, objects in types.items():
            html_content += f'<li><span class="caret">{object_type}</span><ul class="nested">'
            for object_name, details in objects.items():
                html_content += generate_object_html(object_name, object_type, details)
            html_content += "</ul></li>"
        html_content += "</ul></li>"

    html_content += """
        </ul>
        <script>
            var toggler = document.getElementsByClassName("caret");
            var i;
            for (i = 0; i < toggler.length; i++) {
                toggler[i].addEventListener("click", function() {
                    this.parentElement.querySelector(".nested").classList.toggle("active");
                    this.classList.toggle("caret-down");
                });
            }
        </script>
    </body>
    </html>
    """
    file_name=f"database_info_tree_{int(time.time())}.html"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(html_content)
    return file_name

def update_database_info(cursor):
    all_info = get_all_information(cursor)
    file_path=generate_html(all_info)
    webbrowser.open_new_tab(file_path)
    print("Database information updated successfully!")







def save_as_text(file_path, config_data):
    try:
        with open(file_path, 'w') as text_file:
            for table_name, table_info in config_data.items():
                text_file.write(f"Table: {table_name}\n")
                text_file.write(f"  Row Count: {table_info['row_count']}\n")
                for column in table_info['columns']:
                    text_file.write(f"  Column: {column['column_name']}, Type: {column['data_type']}, Length: {column['data_length']}\n")
                text_file.write("\n")
    except IOError as e:
        raise IOError(f"Error saving text file: {e}")
 
def delete_old_config(file_path, file_json):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(file_json):
            os.remove(file_json)
    except IOError as e:
        raise IOError(f"Error deleting old configuration file: {e}")
 

def process_database_connection(user_inputs):
    oracle_conn_str = f"{user_inputs['oracle_user']}/{user_inputs['oracle_password']}@//" \
                      f"{user_inputs['oracle_host']}:{user_inputs['oracle_port']}/" \
                      f"{user_inputs['oracle_service_name']}"
 
    delete_old_config(file_path, file_json)
    oracle_conn = cx_Oracle.connect(oracle_conn_str)
 
    cursor = oracle_conn.cursor()
    sql_query_tables = "SELECT table_name FROM user_tables"
    cursor.execute(sql_query_tables)
    tables = cursor.fetchall()
    config_data = {}
    for table_name in tables:
        table_name_upper = table_name[0].upper()
        sql_query_columns = """
            SELECT column_name, data_type, data_length
            FROM all_tab_columns
            WHERE table_name = :1
            ORDER BY column_id
        """
        cursor.execute(sql_query_columns, [table_name_upper])
        columns = cursor.fetchall()
        columns_info = [{'column_name': col_name.upper(), 'data_type': data_type.upper(), 'data_length': data_length} for col_name, data_type, data_length in columns]
        sql_query_row_count = f"SELECT COUNT(*) FROM {table_name_upper}"
        cursor.execute(sql_query_row_count)
        row_count = cursor.fetchone()[0]
        config_data[table_name_upper] = {
            'columns': columns_info,
            'row_count': row_count
        }
    save_as_text(file_path, config_data)
    save_config(file_json, config_data)
    count_query = "SELECT COUNT(*) FROM user_tables"
    cursor.execute(count_query)
    count = cursor.fetchone()[0]
    chain = create_qa_chain()
    return count, chain, oracle_conn, cursor
 
def create_qa_chain():

        GOOGLE_API_KEY = st.secrets["GEMINI_KEY"]
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
        prompt_template = """
        You are given comprehensive data about various object types in a database. The object types include tables, views, sequences, synonyms, dblinks, indexes, packages, procedures, functions, and triggers. Each object type has specific details:
            Tables: schema contains column_name, data_type, data_length, and row_count for each column. Each table's information is nested under the key 'Table', and each column within that table is listed under the key 'column'.
        {context}

        For any user query, provide a response based on the complete dataset. If the query specifically mentions a table, include all data types and describe the relevant information. Ensure that responses cover all object types and address all questions related to the database comprehensively.        
        Question: {question}
        Helpful Answer:"""
        # embedding_function = HuggingFaceEmbeddings()
        # loader = TextLoader(file_path=file_path)
        # documents = loader.load()
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        # esops_documents = text_splitter.transform_documents(documents)



        embedder=HuggingFaceEmbeddings()
        # vectorstore = FAISS.from_documents(esops_documents, embedder)
        vectorstore=FAISS.load_local("genie_index", embedder, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template) # prompt_template defined above
        llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, callbacks=None, verbose=False)
        document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="Context:\ncontent:{page_content}\nsource:{source}",
            )
        combine_documents_chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_variable_name="context",
                document_prompt=document_prompt,
                callbacks=None,
            )
        qa = RetrievalQA(
                combine_documents_chain=combine_documents_chain,
                retriever=retriever,
                return_source_documents=True,
            )

        return qa





def migrate_dbLinks(oracle_connection, pg_conn, schema_name):
    oracle_cursor=oracle_connection.cursor()
    pg_cursor=pg_conn.cursor()

    # Fetch all dblink information from Oracle
    logging.info("Oracle query to fetch all dblink details", f"""
        SELECT HOST, USERNAME, DB_LINK
        FROM DBA_DB_LINKS
        WHERE OWNER = '{schema_name}'
    """, '\n')
    
    oracle_cursor.execute(f"""
        SELECT HOST, USERNAME, DB_LINK
        FROM DBA_DB_LINKS
        WHERE OWNER = '{schema_name}'
    """)
    
    dblinks_info = oracle_cursor.fetchall()
    logging.info(dblinks_info)    
    if dblinks_info:
        for dblink_info in dblinks_info:
            host, username, db_link = dblink_info
            logging.info(f"Creating db_link: {db_link}")
            st.write(f"Creating db_link: {db_link}")
            logging.info("Creating foreign data wrapper in PostgreSQL")
            # Create foreign data wrapper and server in PostgreSQL
            pg_cursor.execute("""
                CREATE EXTENSION IF NOT EXISTS postgres_fdw;
            """)
            
            logging.info("Creating server in PostgreSQL using: ", f"""
                CREATE SERVER IF NOT EXISTS {db_link}_server
                FOREIGN DATA WRAPPER postgres_fdw
                OPTIONS (host '{host}', dbname 'aidb');
            """)
            pg_cursor.execute(f"""
                CREATE SERVER IF NOT EXISTS {db_link}_server
                FOREIGN DATA WRAPPER postgres_fdw
                OPTIONS (host '{host}', dbname 'aidb');
            """)
            
            logging.info("Creating mapping using query: ", f"""
                CREATE USER MAPPING IF NOT EXISTS FOR CURRENT_USER
                SERVER {db_link}_server
                OPTIONS (user '{username}', password 'remote_db_password');
            """)
            pg_cursor.execute(f"""
                CREATE USER MAPPING IF NOT EXISTS FOR CURRENT_USER
                SERVER {db_link}_server
                OPTIONS (user '{username}', password 'remote_db_password');
            """)
            
            st.write(f"Foreign data wrapper for {db_link} created successfully in PostgreSQL.")
    else:
        st.write(f"No DBLinks found for user '{schema_name}' in Oracle database.")
    
    # Commit changes in PostgreSQL
    pg_conn.commit()

