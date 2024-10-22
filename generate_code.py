from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
import pandas as pd

llm_hub= HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",huggingfacehub_api_token=st.secrets["HUGGINGFACE_KEY"])
# llm_hub=ChatOllama(model="mistral:latest", temperature=0.2)
code=""" import cx_Oracle
import psycopg2
def get_oracle_schema_info(oracle_conn_str):
try:
    oracle_conn = cx_Oracle.connect(oracle_conn_str)
    cursor = oracle_conn.cursor()
 
    schema_info = {}
    cursor.execute("SELECT table_name FROM user_tables")
    tables = cursor.fetchall()
 
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT column_name, data_type, data_length FROM user_tab_columns WHERE table_name = '{table_name}'")
        columns = cursor.fetchall()
        schema_info[table_name] = columns
 
    cursor.close()
    oracle_conn.close()
    return schema_info
 
except cx_Oracle.DatabaseError as e:
    st.write(f"Error: {e")
    return None
   
def migrate_schema_to_postgresql(postgres_conn_str, schema_info):
try:
    postgres_conn = psycopg2.connect(postgres_conn_str)
    cursor = postgres_conn.cursor()
 
    for table_name, columns in schema_info.items():
        create_table_sql = f"CREATE TABLE {table_name ("
        column_defs = []
 
        for column in columns:
            column_name, data_type, data_length = column
            if data_type == "VARCHAR2":
                st.write("Loop1")
                column_defs.append(f"{column_name VARCHAR({data_length})")
            elif data_type == "NUMBER":
                st.write("Loop2")
                column_defs.append(f"{column_name} NUMERIC")
            elif data_type == "DATE":
                st.write("Loop3")
                column_defs.append(f"{column_name} DATE")
            else:
                column_defs.append(f"{column_name} {data_type}")
 
        create_table_sql += ", ".join(column_defs) + ");"
        st.write(create_table_sql)
        cursor.execute(create_table_sql)
        st.write(f"Table {table_name} created successfully.")
        st.write(f"columns{columns}created sucessfully")
 
    postgres_conn.commit()
    cursor.close()
    postgres_conn.close()
 
except psycopg2.DatabaseError as e:
    st.write(f"Error: {e}")
if __name__ == "__main__":
 
oracle_conn_str = "use_name/password@//localhost:1523/xepdb1"
postgres_conn_str = "dbname= user= password= host=localhost port=5432"
 
schema_info = get_oracle_schema_info(oracle_conn_str)
if schema_info:
    migrate_schema_to_postgresql(postgres_conn_str, schema_info)
    """

prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""You are an assistant. If the user asks you to generate the code related to migration from oracle to postgresql  or any other databases in any language.
    based on the database explain the each and every process.
    This is example how you can generate the code like this {code}.
 
    """
)
 
def generate_migration_code(chat_input):
    # Create the LLM chain
    chain = LLMChain(llm=llm_hub, prompt=prompt)
 
    # Format the template with user input and code
    response = chain.invoke(chat_input)
   
    return response

