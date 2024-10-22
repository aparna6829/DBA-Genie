import cx_Oracle
import psycopg2
from psycopg2 import sql
import json
import streamlit as st
import time
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
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








def display_validation(oracle_counts,postgres_counts):
    data = {
        'Object Type': ['Tables', 'Views', 'Indexes', 'Functions', 'Procedures', 'Triggers','Sequences','DatabaseLinks'],
        'Oracle Counts': [oracle_counts.get('tables', 0), oracle_counts.get('views', 0), oracle_counts.get('indexes', 0),
                        oracle_counts.get('functions', 0), oracle_counts.get('procedures', 0),oracle_counts.get('triggers', 0), oracle_counts.get('sequences', 0),oracle_counts.get('dblinks', 0)],
        'PostgreSQL Counts': [postgres_counts.get('tables', 0), postgres_counts.get('views', 0), postgres_counts.get('indexes', 0),
                            postgres_counts.get('functions', 0), postgres_counts.get('procedures', 0), postgres_counts.get('triggers', 0), postgres_counts.get('sequences', 0),postgres_counts.get('dblinks', 0)]  # PostgreSQL does not have DB Links
    }

    df = pd.DataFrame(data)

    # Display the DataFrame in Streamlit
    st.dataframe(df)

    
    
def get_oracle_table_counts(schema_name,oracle_conn_str):
    dsn = cx_Oracle.makedsn('promptoraserver', port=1521, service_name='xepdb1')
    oracle_connection = cx_Oracle.connect(oracle_conn_str)
    oracle_cursor = oracle_connection.cursor()
    
    oracle_query = f"""
    SELECT table_name, num_rows
    FROM all_tables
    WHERE owner = '{schema_name}'
    """
    
    oracle_cursor.execute(oracle_query)
    oracle_tables_info = oracle_cursor.fetchall()
    
    oracle_cursor.close()
    oracle_connection.close()
    
    return pd.DataFrame(oracle_tables_info, columns=['Table Name', 'Row Count (Oracle)'])

# Function to get PostgreSQL table row counts
def get_postgres_table_counts(schema_name,postgres_conn_str):
    postgres_connection = psycopg2.connect(postgres_conn_str)
    postgres_cursor = postgres_connection.cursor()
    
    table_query = f"""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = '{schema_name}'
    """
    
    postgres_cursor.execute(table_query)
    postgres_tables = postgres_cursor.fetchall()
    
    postgres_table_counts = {}
    exclude=['geography_columns','geometry_columns','spatial_ref_sys']
    for (table_name,) in postgres_tables:
        if table_name.lower() not in exclude:
            # print(table_name)
            count_query = f"""
            SELECT COUNT(*) FROM "{schema_name}"."{table_name.upper()}"
            """
            # print(count_query)
            postgres_cursor.execute(count_query)
            count = postgres_cursor.fetchone()[0]
            postgres_table_counts[table_name] = count
    
    postgres_cursor.close()
    postgres_connection.close()
    
    return pd.DataFrame(list(postgres_table_counts.items()), columns=['Table Name', 'Row Count (PostgreSQL)'])


def merge_table_counts(oracle_df, postgres_df):
    merged_df = pd.merge(oracle_df, postgres_df, on='Table Name', how='outer').fillna(0)
    return merged_df


def measure_query_time(query, connection_string, db_type):
    start_time = time.time()
    if db_type == 'oracle':
        connection = cx_Oracle.connect(connection_string)
    elif db_type == 'postgres':
        connection = psycopg2.connect(connection_string)
    else:
        raise ValueError("Unsupported database type")
    cursor = connection.cursor()
    cursor.execute(query)
    cursor.fetchall()  # Fetch all results to ensure the query is fully executed
    end_time = time.time()
    connection.close()
    return end_time - start_time

def timelines(oracle_connection_string,postgres_connection_string):
    query = st.text_area("Enter SQL Query:", 'SELECT * FROM "HR"."COUNTRIES"')
    if st.button("Measure Query Time"):
        with st.spinner('Measuring query time...'):
            # Measure time for Oracle
            try:
                oracle_time = measure_query_time(query.upper(), oracle_connection_string, 'oracle')
                st.success(f"Oracle query time: {oracle_time:.2f} seconds")
            except Exception as e:
                st.error(f"Error measuring Oracle query time: {e}")
            
            # Measure time for PostgreSQL
            try:
                postgres_time = measure_query_time(query.upper(), postgres_connection_string, 'postgres')
                st.success(f"PostgreSQL query time: {postgres_time:.2f} seconds")
            except Exception as e:
                st.error(f"Error measuring PostgreSQL query time: {e}")


def query_database002(user_inputs,user_query):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=st.secrets["GEMINI_KEY"])
    input_parser_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=st.secrets["GEMINI_KEY"])
    # Set up the Oracle database connection
    connection_string = f"oracle+cx_oracle://{user_inputs['oracle_user']}:{user_inputs['oracle_password']}@{user_inputs['oracle_host']}:{user_inputs['oracle_port']}/?service_name={user_inputs['oracle_service_name']}"
    temp_conn = cx_Oracle.connect(f"{user_inputs['oracle_user']}/{user_inputs['oracle_password']}@//{user_inputs['oracle_host']}:{user_inputs['oracle_port']}/{user_inputs['oracle_service_name']}")

    # Create SQLDatabase instance
    db = SQLDatabase.from_uri(connection_string)

    # Create SQLDatabaseToolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Define the SQL query template
    sql_query_template = """
    You are the system DBA of an Oracle server. You can query the all_tables, all_objects tables to find any object. Query the Oracle database and retrieve relevent information according to the following user query: {user_query}. 
    After retrieving the data, summarize the results in a clear and concise natural language response. Ensure that the explanation is understandable, highlights key findings, and provides any relevant context or insights."""

    # Initialize the main agent
    agent = initialize_agent(
        toolkit.get_tools(),
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )
    sql_query = sql_query_template.format(
        user_query=user_query
    )    
    agent.run(sql_query)


def identify_oracle_only_objects(merged_df):
    oracle_only = merged_df[merged_df['Row Count (PostgreSQL)'] == 0]
    return oracle_only[['Table Name', 'Row Count (Oracle)']]




def getall_oracle_objects(schema_name, oracle_conn_str):
    oracle_connection = cx_Oracle.connect(oracle_conn_str)
    oracle_cursor = oracle_connection.cursor()
    
    oracle_query = f"""
    SELECT object_name, object_type
    FROM all_objects
    WHERE owner = '{schema_name}'
    """
    
    oracle_cursor.execute(oracle_query)
    oracle_objects = oracle_cursor.fetchall()
    # print(oracle_objects)
    
    oracle_cursor.close()
    oracle_connection.close()
    
    return pd.DataFrame(oracle_objects, columns=['Object Name', 'Object Type'])

# Function to get PostgreSQL objects
def getall_postgres_objects(schema_name, postgres_conn_str):
    postgres_connection = psycopg2.connect(postgres_conn_str)
    postgres_cursor = postgres_connection.cursor()
    
    postgres_query = f"""
    SELECT table_name as object_name, 'TABLE' as object_type
    FROM information_schema.tables
    WHERE table_schema = '{schema_name}'
    
    UNION ALL
    
    SELECT table_name as object_name, 'VIEW' as object_type
    FROM information_schema.views
    WHERE table_schema = '{schema_name}'
    
    UNION ALL
    
    SELECT indexname as object_name, 'INDEX' as object_type
    FROM pg_indexes
    WHERE schemaname = '{schema_name}'
    
    UNION ALL
    
    SELECT sequence_name as object_name, 'SEQUENCE' as object_type
    FROM information_schema.sequences
    WHERE sequence_schema = '{schema_name}'
    
    UNION ALL
    
    SELECT routine_name as object_name, 'FUNCTION' as object_type
    FROM information_schema.routines
    WHERE routine_schema = '{schema_name}'
    AND routine_type = 'FUNCTION'
    

    UNION ALL
    SELECT routine_name as object_name, 'PROCEDURE' as object_type
        FROM information_schema.routines
        WHERE routine_schema = '{schema_name}'
        AND routine_type = 'PROCEDURE'

    UNION ALL
    SELECT trigger_name AS object_name, 'TRIGGER' AS object_type
    FROM information_schema.triggers
    WHERE trigger_schema = '{schema_name}'
    """
    
    postgres_cursor.execute(postgres_query)
    postgres_objects = postgres_cursor.fetchall()
    
    postgres_cursor.close()
    postgres_connection.close()
    
    return pd.DataFrame(postgres_objects, columns=['Object Name', 'Object Type'])

# Function to identify and print objects left out in Oracle
def identify_oracle_only_objects1(oracle_df, postgres_df):
    merged_df = pd.merge(oracle_df, postgres_df, on=['Object Name', 'Object Type'], how='left', indicator=True)
    oracle_only = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    return oracle_only


