import streamlit as st
from langchain.schema import HumanMessage, AIMessage
import json 
import time
import os
import sys
import cx_Oracle
import psycopg2
import logging
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from generate_code import generate_migration_code
import streamlit as st
from libraries3 import migrate_dbLinks,extract_db_details, extract_postgres_db_details, default_serializer, save_config,update_database_info, save_as_text, delete_old_config, process_database_connection, create_qa_chain
from schema_tables import create_schema_and_user, extract_columns_to_config, analyze_datatypes, get_llm_suggestion, migrate_table_structure, migrate_table_data, migrate_table_range, migrate_schema_structure, migrate_schema_data, get_oracle_users, get_the_Schema_name_from_user, get_object_query, generate_html_report, get_schema_info, get_object_details, estimate_cost, get_all_object_details, setup, generate_migration_summary, parse_user_input, generate_llm_insights, save_detailed_results, open_file, get_oracle_table_details, get_postgres_table_details
from index import migrate_schema_objects_fun, migrate_schema_objects, Load_LLM, get_user_indexes, get_oracle_object_details,migrate_triggers, migrate_indexes, get_postgres_object_details, save_indexes_to_config, load_indexes_from_config, check_index_exists, check_table_exists, migrate_index, fetch_solutions_for_errors, create_schema, migrate_sequence, seq_syn_migrate, migrate_synonym, error_messages, llm_validation, validating, view_migration, extract_tables_from_view,convert_decode_to_case,convert_oracle_to_postgres,migrate_view,check_postgres_table_exists,get_oracle_view,get_oracle_views_in_schema
from langchain.schema import HumanMessage, AIMessage
import json 
from tables import extract_columns_to_config1, migrate_table_structures, migrate_data, get_system_info, calculate_dynamic_values, load_or_create_migrated_tables, migrate_complete_data, get_oracle_object_counts, get_postgres_object_counts
from validation import getall_oracle_objects,getall_postgres_objects,identify_oracle_only_objects1,identify_oracle_only_objects,merge_table_counts,display_validation, get_oracle_table_counts, get_postgres_table_counts, timelines,query_database002
import os
from langchain_community.llms import LlamaCpp
import cx_Oracle
import pandas as pd
import webbrowser
import psycopg2
import logging
import streamlit.components.v1 as components
error_messages=[]



st.set_page_config(page_title="Migrations Bot",
layout="wide")



header = st.container()
header.title("DBAGenie")
header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
st.markdown(
    """
<style>
    div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
        position: sticky;
        top: 0rem;
        background: linear-gradient(to right, #c8ae94, #3eacda);
        z-index: 999;
        text-align:center;
        color:white;
       
       
 
 
</style>
    """,
    unsafe_allow_html=True
)


log_filename = f"migration_log.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)
load_css()
if "messages" not in st.session_state:
    st.session_state.messages = []

if "connections_submitted" not in st.session_state:
    st.session_state.connections_submitted = False
if "metadata_extracted" not in st.session_state:
    st.session_state.metadata_extracted = False
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = {}
if "postgres_input" not in st.session_state:
    st.session_state.postgres_input = {}
if "chat_history" not in st.session_state:
      st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a coder assistant. Ask me anything about code."),
      ]
if "chat_history_db" not in st.session_state:
    st.session_state.chat_history_db = [
        AIMessage(content="Hello! I'm an Oracle assistant. Ask me anything about your database."),
    ]
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8=st.tabs(["DBA_Genie 🧞","Connection Details 🔗","Discovery 🕵🏻‍♂️","Assessment 📝","Migration 🔄","Validation ✅","Big_Data Query 📉","Generate Migration Code 👩🏻‍💻"])
with tab1:
    container=st.container(height=400)
    if "chat_history_dba" not in st.session_state:
        st.session_state.chat_history_dba = [
        AIMessage(content="Hello! I'm a DBAGeine assistant. Ask me anything about DBAGeine."),
        ]
    with st.container(height=420,border=0):
        llm = ChatOpenAI(model='gpt-4o', api_key=st.secrets["OPENAIAPI_KEY"])
        embedding_path=r"C:\github\GenAI\DBA-Genie\DBA"
        # Initialize the language model
        embeddings = HuggingFaceEmbeddings()
        
        docsearch = FAISS.load_local(embedding_path, embeddings,allow_dangerous_deserialization=True)
        # llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", return_source_documents=True, retriever=docsearch.as_retriever())
        if 'user_input' not in st.session_state:
            st.session_state.user_input = ""
        # Streamlit layout
        with st.container(height=100):
            col1, spacer1, col2, spacer2, col3, spacer3, col4 = st.columns([1, 0.04, 1, 0.04 ,1.5, 0.04, 1])
       
            with col1:
                if st.button("What is meant by DBAGenie?"):
                    st.session_state.user_input = "What is meant by DBAGenie?"

            with col2:
                if st.button("What are the Benifits of DBAGenie?"):
                    st.session_state.user_input = "What are the Benifits of DBAGenie?"

            with col3:
                if st.button("What are the target and Source databases Supported by DBAGeine?"):
                    st.session_state.user_input = "What are the target and Source databases Supported by DBAGeine?"
                    
            with col4:
                if st.button("How DBAGenie Leverages Generative AI?"):
                    st.session_state.user_input = "How DBAGenie Leverages Generative AI?"
            # # User text input
        with container:
            for message in st.session_state.chat_history_dba:
                if isinstance(message, AIMessage):
                    with st.container():
                        with st.chat_message("AI"):
                            st.markdown(message.content)
                elif isinstance(message, HumanMessage):
                    with st.container():
                        with st.chat_message("Human"):
                            st.markdown(message.content)
            text_input = st.chat_input("Enter your question here:")

            if text_input:
                st.session_state.user_input = text_input

            # Process the user input
            if st.session_state.user_input:
                st.session_state.chat_history_dba.append(HumanMessage(content=st.session_state.user_input))
                with st.container():
                    with st.chat_message("Human"):
                        st.markdown(st.session_state.user_input)
                with st.spinner("Processing"):
                    response = chain.invoke(st.session_state.user_input)
                    result=(response['answer'])
                    
                    logging.info(result)
                    with st.container():
                        with st.chat_message("AI"):
                            st.markdown(result)
                    st.session_state.chat_history_dba.append(AIMessage(content=result))
with tab2:
    st.header("Database Connection Details")

    # Predefined connection prompt
    connection_prompt = st.text_area("Enter your Oracle Connection Details:", key="test")

    # Separate text input for the password
    oracle_password = st.text_input("Oracle Password", type="password",key="test11")

    postgres_prompt = st.text_area("Enter your Postgres Connection Details:", key="postgres")

    # tab1,tab2,tab3,tab4=st.tabs(["Discovery","Assessment","Migration","Validation"])
    # Separate text input for the password
    postgres_password = st.text_input("Postgres Password", type="password",key="test12")
    if st.button("Submit Details:"):
        extracted_oracle_details = extract_db_details(connection_prompt)
        if extracted_oracle_details:
            # Map the extracted details to the format expected by process_database_connection
            st.session_state.user_inputs = {
                "oracle_user": extracted_oracle_details.get("Oracle Username"),
                "oracle_password": oracle_password,
                "oracle_host": extracted_oracle_details.get("Oracle Host"),
                "oracle_port": extracted_oracle_details.get("Oracle Port"),
                "oracle_service_name": extracted_oracle_details.get("Oracle Service Name")
            }
            oracle_conn_str = f"{st.session_state.user_inputs['oracle_user']}/{st.session_state.user_inputs['oracle_password']}@//{st.session_state.user_inputs['oracle_host']}:{st.session_state.user_inputs['oracle_port']}/{st.session_state.user_inputs['oracle_service_name']}"
            try:
                cx_Oracle.connect(oracle_conn_str)
                connected_to_oracle = True
            except:
                st.error(f"Please check again and enter valid Oracle database details")
            # process_database_connection(st.session_state.user_inputs)
        extracted_postgres_details = extract_postgres_db_details(postgres_prompt)
        if extracted_postgres_details:
            # Map the extracted details to the format expected by process_database_connection
            st.session_state.postgres_input = {
                "postgres_user": extracted_postgres_details.get("username"),
                "postgres_password": postgres_password,
                "postgres_host": extracted_postgres_details.get("host"),
                "postgres_port": extracted_postgres_details.get("port"),
                "postgres_dbname": extracted_postgres_details.get("dbname")
            }
            postgres_conn_str = f"dbname={st.session_state.postgres_input['postgres_dbname']} user={st.session_state.postgres_input['postgres_user']} password={st.session_state.postgres_input['postgres_password']} host={st.session_state.postgres_input['postgres_host']} port={st.session_state.postgres_input['postgres_port']}"
            try:
                psycopg2.connect(postgres_conn_str)
                connected_to_postgres = True
            except:
                st.error(f"Please check again and enter valid PostgreSQL details")
        try:
            if connected_to_oracle and connected_to_postgres:
                st.session_state.connections_submitted = True
                st.write("Extracted Details:")
                for key, value in st.session_state.user_inputs.items():
                    if key == "oracle_password":
                        value = "*" * len(value)  # Mask the password based on its length
                    st.write(f"{key}: {value}")
                for key, value in st.session_state.postgres_input.items():
                    if key == "postgres_password":
                        value = "*" * len(value)  # Mask the password based on its length
                    st.write(f"{key}: {value}")
                
                st.success("Connection details extracted and submitted successfully.")
            
            else:
                st.error("Failed to extract connection details. Please check your input.")
        except Exception as e:
            st.write(f"Following error occured: {e}")


with tab3:
    if st.session_state.connections_submitted:
        if st.button('Generate Tree Structure'):
            oracle_connection = cx_Oracle.connect(f"{st.session_state.user_inputs['oracle_user']}/{st.session_state.user_inputs['oracle_password']}@{st.session_state.user_inputs['oracle_host']}:{st.session_state.user_inputs['oracle_port']}/{st.session_state.user_inputs['oracle_service_name']}")
            cursor=oracle_connection.cursor()
            update_database_info(cursor)
            st.write("Database tree view generated successfully!")
            # Define the path to your HTML file
            
            
        count, chain, oracle_conn, cursor = process_database_connection(st.session_state.user_inputs)
        # st.write(f":blue[There are a total of {count} tables in the database]")
        for message in st.session_state.chat_history_db:
            with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
                st.markdown(message.content)
        query = st.chat_input("Enter your input here:", key="test1")
        if query:
            
            st.session_state.chat_history_db.append(HumanMessage(content=query))
            with st.chat_message("Human"):
                st.markdown(query)
            with st.spinner("Processing"):
                response=chain.invoke({'query':query})
                result = response['result']
                with st.chat_message("AI"):
                    st.markdown(result)
                st.session_state.chat_history_db.append(AIMessage(content=result))
    else:
        st.warning("Please submit database connection details in the sidebar first.")
with tab4:
    if st.session_state.connections_submitted:
        count, chain, oracle_conn, cursor = process_database_connection(st.session_state.user_inputs)
        oracle_conn_str = f"{st.session_state.user_inputs['oracle_user']}/{st.session_state.user_inputs['oracle_password']}@//" \
                            f"{st.session_state.user_inputs['oracle_host']}:{st.session_state.user_inputs['oracle_port']}/" \
                            f"{st.session_state.user_inputs['oracle_service_name']}"

        sga_size, cpu_count, available_memory = get_system_info(oracle_conn_str)

        st.session_state.MAX_WORKERS, st.session_state.BATCH_SIZE, st.session_state.CHUNK_SIZE = calculate_dynamic_values(sga_size, cpu_count, available_memory)
        with st.expander("System Configuration"):
            st.write(f"SGA_SIZE: {sga_size}\nCPU COUNT: {cpu_count}\nAvailable Memory: {available_memory}")
            st.write(f"Dynamic settings: MAX_WORKERS={st.session_state.MAX_WORKERS}, BATCH_SIZE={st.session_state.BATCH_SIZE}, CHUNK_SIZE={st.session_state.CHUNK_SIZE}")
        if cpu_count<=4:
            st.write("LLM Suggestion: Increase the CPU cores for better speeds")
        if cpu_count>10:
            st.write("LLM Suggestion: CPU cores are enough")
        if "schema_names_assessment" not in st.session_state:
            st.session_state.schema_names_assessment=[]
        users = get_oracle_users(cursor)
        with st.expander("User Schemas(Non-Defualt User Schemas)"):
            st.write(users)
        schema_name=st.text_input("Enter the Oracle schema/schemas(seperated by a comma) to generate the assessment report. Type \'ALL\' to generate the assessment report for all the schemas under the user:")
        if schema_name:
                    if schema_name.strip().upper()=='ALL':
                        st.session_state['schema_names_assessment']=[user for user in users]
                    # Get schema names from user input
                    else:
                        schema_names = schema_name.strip().split(',')
                        # Save schema names in session state
                        st.session_state['schema_names_assessment'] = [schema_name.strip() for schema_name in schema_names]
        if st.button("Generate Report"):
            for schema_name in st.session_state['schema_names_assessment']:
                if schema_name in users:
                    try:
                        object_type_query=get_object_query()
                        schema_info = get_schema_info(cursor,schema_name)
                        print(schema_info)
                        cursor.execute(object_type_query, owner=schema_name)
                        object_types = cursor.fetchall()

                        object_data = {}
                        for object_type, _ in object_types:
                            st.write(f"Processing {object_type}...")
                            details = get_object_details(cursor, schema_info['schema'],  object_type_query, object_type)
                            if details:
                                object_data[object_type] = details
                        object_data = generate_llm_insights(object_data)

                        # Save detailed results for verification
                        save_detailed_results(object_data, "detailed_migration_results.json")

                        # Generate and save HTML report
                        html_content = generate_html_report(schema_info, object_data)
                        file_name=f"oracle_to_postgres_migration_report_{int(time.time())}.html"
                        with open(file_name, "w") as file:
                            file.write(html_content)
                        st.write("Migration report has been created successfully.")
                        st.write("Detailed results saved in 'detailed_migration_results.json'")
                        with st.spinner("Opening report"):
                            open_file(file_name)
                    except Exception as e:
                        st.warning(f"The following exception occured for the schema: {schema_name}, Error: {e}")
                else:
                    st.warning("Enter a valid schema name")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        sql_query_template, agent, input_parser_agent, temp_conn=setup(st.session_state.user_inputs)
        user_input = st.text_area("Enter your query (e.g., 'Analyze table EMPLOYEES and suggest migration steps')",key="tab2")
        if st.button("Generate Migration Summary"):
            if user_input:
                with st.spinner("Parsing input and generating migration summary..."):
                    
                    # Parse user input
                    parsed_input = parse_user_input(user_input, input_parser_agent)
                    st.write("Parsed Input:", parsed_input)
                    # print('\n\n\n',parsed_input,'\n\n')
                    # Extract object type, object name, and user query from parsed input
                    try:
                        lines = parsed_input.split('\n')
                        # print('\n\n',lines,'\n\n')
                        object_type = lines[0].split(': ')[1]
                        object_name = lines[1].split(': ')[1]
                        user_query = lines[2].split(': ')[1] if lines[2].split(': ')[1] != 'None' else ''
                        # print(lines,'\n\n\n')
                    except:
                        lines = parsed_input.split(',')
                        # print('\n\n',lines,'\n\n')
                        object_type = lines[0].split(': ')[1]
                        object_name = lines[1].split(': ')[1]
                        user_query = lines[2].split(': ')[1] if lines[2].split(': ')[1] != 'None' else ''
                        # print(lines,'\n\n\n')
                        
                    # Generate migration summary
                    summary = generate_migration_summary(object_type, object_name, user_query,  sql_query_template, agent, temp_conn)
                    
                    # Add user query to chat history
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": summary})
                    
                    # Display the latest response
                    with st.chat_message("assistant"):
                        st.markdown(summary)
            else:
                st.warning("Please enter a query.")
            #     st.error(f"Failed loading file. Using cached file instead..")
            #     st.write("Opening file")
            #     webbrowser.open(r"oracle_to_postgres_migration_report.html")
    else:
        st.warning("Please submit database connection details in the sidebar first.")
with tab5:
    if st.session_state.connections_submitted:
        try:
            oracle_connection = cx_Oracle.connect(f"{st.session_state.user_inputs['oracle_user']}/{st.session_state.user_inputs['oracle_password']}@{st.session_state.user_inputs['oracle_host']}:{st.session_state.user_inputs['oracle_port']}/{st.session_state.user_inputs['oracle_service_name']}")
            pg_connection = psycopg2.connect(f"dbname={st.session_state.postgres_input['postgres_dbname']} user={st.session_state.postgres_input['postgres_user']} password={st.session_state.postgres_input['postgres_password']} host={st.session_state.postgres_input['postgres_host']} port={st.session_state.postgres_input['postgres_port']}")
            oracle_cursor = oracle_connection.cursor()
            pg_cursor = pg_connection.cursor()
            users = get_oracle_users(oracle_cursor)
            with st.expander("User Schemas(Non-Default User Schemas)"):
                schema_name = st.write(users)
            if st.session_state.connections_submitted:
                oracle_conn_str = f"{st.session_state.user_inputs['oracle_user']}/{st.session_state.user_inputs['oracle_password']}@//" \
                            f"{st.session_state.user_inputs['oracle_host']}:{st.session_state.user_inputs['oracle_port']}/" \
                            f"{st.session_state.user_inputs['oracle_service_name']}"
                postgres_conn_str=f"dbname={st.session_state.postgres_input['postgres_dbname']} user={st.session_state.postgres_input['postgres_user']} password={st.session_state.postgres_input['postgres_password']} host={st.session_state.postgres_input['postgres_host']} port={st.session_state.postgres_input['postgres_port']}"
                if 'schema_names' not in st.session_state:
                    st.session_state['schema_names'] = []
                # Get user input
                chat_input = st.text_input("Enter the Oracle schema/schemas(seperated by a comma) to migrate the schemas. Type \'ALL\' to migrate all the schemas under the user:",key="Migration Schema")
                if chat_input:
                    
                    if chat_input.strip().upper()=='ALL':
                        st.session_state['schema_names']=[user for user in users]
                    # Get schema names from user input
                    else:
                        response = get_the_Schema_name_from_user(chat_input)
                        schema_names = response.content.strip().split(',')
                        # Save schema names in session state
                        st.session_state['schema_names'] = [schema_name.strip() for schema_name in schema_names]
                if st.button("Start Migration"):
                # Display and use schema names
                    for schema_name in st.session_state['schema_names']:
                        oracle_connection = cx_Oracle.connect(f"{st.session_state.user_inputs['oracle_user']}/{st.session_state.user_inputs['oracle_password']}@{st.session_state.user_inputs['oracle_host']}:{st.session_state.user_inputs['oracle_port']}/{st.session_state.user_inputs['oracle_service_name']}")
                        pg_connection = psycopg2.connect(f"dbname={st.session_state.postgres_input['postgres_dbname']} user={st.session_state.postgres_input['postgres_user']} password={st.session_state.postgres_input['postgres_password']} host={st.session_state.postgres_input['postgres_host']} port={st.session_state.postgres_input['postgres_port']}")
                        oracle_cursor = oracle_connection.cursor()
                        pg_cursor = pg_connection.cursor()
                        key=st.session_state['schema_names'].index(schema_name)
                        if schema_name:
                            st.header(f"Performing migration for the schema :orange[{schema_name}]")
                            st.write(":blue[Migrating Table Schemas]")
                            config = extract_columns_to_config1(schema_name,oracle_conn_str)
                            migrated_tables, special_tables = migrate_table_structures(schema_name, config, oracle_conn_str,postgres_conn_str)
                            st.write(f"Migrated tables: {migrated_tables}")
                            # st.write(f"Tables with special types (skipped): {special_tables}")
                            st.write(f":green[Successfully migrated the tables for schema {schema_name}]")
                            
                            
## Migrating VIEWS
                            
                            st.divider()
                            st.header(":blue[Migrating Views]")
                            with st.spinner("Migrating Views..."):
                                
                                try:
                                    oracle_conn = cx_Oracle.connect(oracle_conn_str)
                                    pg_conn=psycopg2.connect(postgres_conn_str)
                                    with st.expander("Views Migration"):
                                        view_migration(oracle_conn,pg_conn,schema_name)
                                    st.write("Migrated Views Successfully")

                                finally:
                                    oracle_conn.commit()
                                    oracle_conn.close()
                                    pg_connection.commit()
                                    pg_connection.close()


## Migrating DBLinks
                            st.divider()
                            st.header(":blue[Migrating Database Links]")
                            with st.spinner("Migrating DBLinks"):
                                with st.expander("Database Links Summary"):
                                    try:
                                        oracle_conn = cx_Oracle.connect(oracle_conn_str)
                                        pg_conn=psycopg2.connect(postgres_conn_str)
                                        oracle_cursor=oracle_conn.cursor()
                                        pg_cursor=pg_conn.cursor()
                                        migrate_dbLinks(oracle_connection, pg_conn, schema_name)
                                        
                                    except (cx_Oracle.Error, psycopg2.Error) as error:
                                        st.write("Error occurred:", error)

                                    finally:
                                        # Close connections
                                        if oracle_cursor:
                                            oracle_cursor.close()
                                        if oracle_conn:
                                            oracle_conn.close()
                                        if pg_cursor:
                                            pg_cursor.close()
                                        if pg_conn:
                                            pg_conn.close()



        # expanders for procedures and functions
                            st.divider()
                            st.header(":blue[Migrating Functions]")
                            with st.spinner("Migrating Functions"):
                                    oracle_conn = cx_Oracle.connect(oracle_conn_str)
                                    pg_connection=psycopg2.connect(postgres_conn_str)
                                    with st.expander("Functions Migration"):
                                        migrate_schema_objects_fun(schema_name,oracle_conn,pg_connection)



                                    # if error_messages:
                                    #     with st.expander("Error Solutions"):
                                    #         st.write("Error solutions here")

                            st.divider()
                            st.header(":blue[Migrating Procedures]")
                            with st.spinner("Migrating Procedures"):    
                                    oracle_conn = cx_Oracle.connect(oracle_conn_str)
                                    pg_connection=psycopg2.connect(postgres_conn_str)
                                    with st.expander("Procedure Migration"):
                                        migrate_schema_objects(schema_name,oracle_conn,pg_connection)
                            # if error_messages:
                            #     solutions = fetch_solutions_for_errors(error_messages)
                            #     with st.expander("Error Solutions"):
                            #         st.write("Errors occurred during migration. Please review the error messages above and consider the following solutions:")
                            #         for error_message, solution in zip(error_messages, solutions):
                            #             st.write(f"**Error:** {error_message}")
                            #             st.write(f"**Solution:** {solution}")
                    
    # Expanders for Triggers
                            # st.divider()
                            # st.header(":blue[Migrating Triggers]")
                            # with st.spinner("Migrating Triggers..."):

                            #     try:
                            #         with st.expander("Triggesr Migration"):
                            #             migrate_triggers(oracle_conn_str,postgres_conn_str,schema_name)
                            #     except Exception as e:
                            #         st.write(f"Fatal error: {e}")
                            #         sys.exit(1)
    # Expanders for Sequences
                            st.divider()
                            st.header(":blue[Migrating Sequences]")
                            with st.spinner("Migrating Sequences..."):
                                try:
                                    oracle_conn = cx_Oracle.connect(oracle_conn_str)
                                    pg_conn = psycopg2.connect(postgres_conn_str)
                                    with st.expander("Sequences Migration"):
                                        seq_syn_migrate(oracle_conn,pg_conn,schema_name,"sequence")
                                    st.write("Migration of Sequences done")
                                finally:
                                    oracle_conn.commit()
                                    oracle_conn.close()
                                    pg_conn.commit()
                                    pg_conn.close()
                            
    # Expanders for synonyms
                            st.divider()
                            st.header(":blue[Migrating Synonyms]") 
                            with st.spinner("Migrating Synonyms..."):
                                try:
                                    oracle_conn = cx_Oracle.connect(oracle_conn_str)
                                    pg_conn = psycopg2.connect(postgres_conn_str)
                                    with st.expander("Synonyms Migration"):
                                        seq_syn_migrate(oracle_conn,pg_conn,schema_name,"synonym")
                                    st.write("Migration of Synonyms done")
                                finally:
                                    oracle_conn.commit()
                                    oracle_conn.close()
                                    pg_conn.commit()
                                    pg_conn.close()

                            
                            st.divider()
                            st.header(":blue[Migrating Table Data]")
                            with st.spinner("Migrating Table Data..."):
                                config_filename = f"{schema_name}_columns_config.json"
                                if not os.path.exists(config_filename):
                                    logging.info(f"Error: {config_filename} not found. Please run the table structure migration script first.")
                                    exit(1)
                                with open(config_filename, "r") as f:
                                    config = json.load(f)
                                with st.expander("Migration Details"):
                                    config_filename = f"{schema_name}_columns_config.json"
                                    if not os.path.exists(config_filename):
                                        print(f"Error: {config_filename} not found. Please run the table structure migration script first.")
                                        exit(1)

                                    with open(config_filename, "r") as f:
                                        config = json.load(f)

                                    migrated_tables = load_or_create_migrated_tables(schema_name, config)

                                    migrate_complete_data(schema_name, config, migrated_tables,oracle_conn_str,postgres_conn_str,st.session_state.MAX_WORKERS,st.session_state.BATCH_SIZE)



                            st.divider()
                            st.header(":blue[Migrating Indexes]")
                            with st.spinner("Migrating indexes..."):
                                try:
                                    oracle_conn = cx_Oracle.connect(oracle_conn_str)
                                    pg_conn=psycopg2.connect(postgres_conn_str)
                                    oracle_cursor = oracle_conn.cursor()
                                    indexes = get_user_indexes(oracle_cursor, schema_name)
                                    print("Fetching indexes")
                                    if not indexes:
                                        st.write(f"No indexes found for user {schema_name} in the Oracle database. Skipping the migration process.")
                                    else:
                                        with st.expander(f"\nIndexes for user {schema_name}:"):
                                            for index in indexes:
                                                st.write(f"{index[0]} (on table {index[1]}, column {index[2]})")
                                        save_indexes_to_config(indexes)

                                        indexes = load_indexes_from_config()

                                        expander=st.expander("Attempting to migrate indexes")
                                        success_count, failure_count, skipped_indexes = migrate_indexes(indexes, pg_conn, schema_name,expander)
                                            
                                        with st.expander("Migration Summary"):
                                            st.write(f"Successfully migrated: {success_count}")
                                            st.write(f"Failed to migrate: {failure_count}")
                                            if skipped_indexes:
                                                st.write("Skipped indexes:")
                                                for idx in skipped_indexes:
                                                    st.write(f"- {idx}")
                                        if failure_count > 0:
                                            with st.expander("\nEncountered errors during the migration process:"):
                                                for error_message in error_messages:
                                                    st.write(error_message)
                                            
                                            # Uncomment the following lines if you want to use the LLM for error solutions
                                            # with st.expander("\nAttempting to provide solutions for the errors:"):
                                            #     solution_prompts = fetch_solutions_for_errors(error_messages)
                                            #     for solution_prompt in solution_prompts:
                                            #         st.write(solution_prompt)
                                        st.write("Index Migration Process Completed")
                                except psycopg2.Error as error:
                                    st.error(f"Error connecting to PostgreSQL database: {error}")
                                except cx_Oracle.Error as error:
                                    st.error(f"Error connecting to Oracle database: {error}")
                                finally:
                                    # Close Oracle connection
                                    oracle_conn.commit()
                                    oracle_conn.close()
                                    pg_conn.commit()
                                    pg_conn.close()


                        else:
                            st.warning("Please enter a schema name.")  


        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.error(error_message)
            error_messages.append(error_message)
    else:
        st.warning("Please submit database connection details in the sidebar first.")

             
            
with tab6:
    st.header("Validation")

    if st.session_state.connections_submitted:
        oracle_conn_str = f"{st.session_state.user_inputs['oracle_user']}/{st.session_state.user_inputs['oracle_password']}@//{st.session_state.user_inputs['oracle_host']}:{st.session_state.user_inputs['oracle_port']}/{st.session_state.user_inputs['oracle_service_name']}"
        postgres_conn_str = f"dbname={st.session_state.postgres_input['postgres_dbname']} user={st.session_state.postgres_input['postgres_user']} password={st.session_state.postgres_input['postgres_password']} host={st.session_state.postgres_input['postgres_host']} port={st.session_state.postgres_input['postgres_port']}"
        oracle_conn=cx_Oracle.connect(oracle_conn_str)
        oracle_cursor=oracle_conn.cursor()
        
        if "schema_names_validation" not in st.session_state:
            st.session_state['schema_names_validation']=[]
        users = get_oracle_users(cursor)
        with st.expander("User Schemas(Non-Defualt User Schemas)"):
            st.write(users)
        schema_name=st.text_input("Enter the Oracle schema/schemas(seperated by a comma) to validate. Type \'ALL\' to vlaidate all the schemas under the user:")
        if schema_name:
            if schema_name.strip().upper()=='ALL':
                st.session_state['schema_names_validation']=[user for user in users]
            # Get schema names from user input
            else:
                schema_names = schema_name.strip().split(',')
                # Save schema names in session state
                st.session_state['schema_names_validation'] = [schema_name.strip() for schema_name in schema_names]
            for schema_name in st.session_state['schema_names_validation']:
                try:
                    if schema_name in users:
                        st.subheader(f"Validation for schema: {schema_name}")
                        oracle_counts = get_oracle_object_counts(schema_name,oracle_conn_str)
                        postgres_counts = get_postgres_object_counts(schema_name,postgres_conn_str)
                        st.write("**Object counts**: ")
                        display_validation(oracle_counts,postgres_counts)
                        oracle_df = get_oracle_table_counts(schema_name,oracle_conn_str)
                        postgres_df = get_postgres_table_counts(schema_name,postgres_conn_str)
                        merged_df = merge_table_counts(oracle_df, postgres_df)
                        oracle_only_objects = identify_oracle_only_objects(merged_df)
                        
                        
                        oracle_df1 = getall_oracle_objects(schema_name, oracle_conn_str)
                        postgres_df1 = getall_postgres_objects(schema_name, postgres_conn_str)

                        # Identify Oracle-only objects
                        oracle_only_objects = identify_oracle_only_objects1(oracle_df1, postgres_df1)

                        # In your Streamlit app
                        with st.expander("Objects that were not migrated/unique to Oracle: "):
                            if not oracle_only_objects.empty:
                                st.subheader('Objects Present in Oracle but Missing in PostgreSQL')
                                st.dataframe(oracle_only_objects)
                            else:
                                st.write("No objects are left out in Oracle.")

                        # In your Streamlit app
                        with st.expander("ROW Count of the Tables"):
                            st.subheader('Oracle and PostgreSQL Table Row Counts')
                            st.dataframe(merged_df)
                        
                        with st.spinner("Validating schema migration..."):
                            validating(schema_name,oracle_conn_str, postgres_conn_str)
                # except Exception as e:
                #     st.warning(f"There is an error: {e}")    
                except Exception as e:
                    st.warning(f"The following exception occured for the schema: {schema_name}, Error: {e}")
        st.header("Compare execution times(_Run queries that can be executed simultaneously in Oracle and Postgres_)")
        timelines(oracle_conn_str,postgres_conn_str)
    else:
        st.warning("Please submit database connection details in the sidebar first.")
with tab7:
    st.write("""Incase you are not automatically redirected to a new tab..""")
    st.markdown('''
            <a href="http://35.225.193.21:8501">
                <button class="nextbutton" style="background-color: lightgrey; border: 0.05px solid black">Click Here</button>
            </a>
            ''',unsafe_allow_html=True)  
with tab8:
    with st.container(height=480,border=0):
        try:
            if st.session_state.connections_submitted:
                for message in st.session_state.chat_history:
                    if isinstance(message, AIMessage):
                        with st.chat_message("AI"):
                            st.markdown(message.content)
                    elif isinstance(message, HumanMessage):
                        with st.chat_message("Human"):
                            st.markdown(message.content)
                logging.info("Generate the Migration code")
                chat_input = st.chat_input("Type your query here:",key="code_generation")
                logging.info(chat_input)
                if chat_input:
                    st.session_state.chat_history.append(HumanMessage(content=chat_input))
                    with st.chat_message("Human"):
                        st.markdown(chat_input)
                    with st.spinner("Processing"):
                        response = generate_migration_code(chat_input)
                        result=(response['text'])
                        
                        logging.info(result)
                        with st.chat_message("AI"):
                            st.markdown(result)
                        st.session_state.chat_history.append(AIMessage(content=result))
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.error(f"An error occurred: {e}", exc_info=True)
            error_messages.append(f"An error occurred: {e}")
        if error_messages:
                    
            for error_message in error_messages:
                
                print(error_message)

        with st.spinner("Fetching Solutions..."):
            with st.expander("Response for Error Message"):
                solutions = fetch_solutions_for_errors(error_messages)
                for i, solution in enumerate(solutions, start=1):
                    # st.write(f"Solution {i}:")
                    st.write(f"```{solution}")
# print("hello")