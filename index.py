import re
import sqlparse
import cx_Oracle
import psycopg2
from psycopg2 import sql
import json
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
error_messages = []
function_errors = []
import logging


log_filename = f"migration_log.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def create_elocation_edge_link_level_function(pg_cursor, pg_schema):
    logging.info("Performing migration of elocation_edge_link_level function")
    check_role_sql = f"""
    SELECT 1 FROM pg_roles WHERE rolname = \'{pg_schema.upper()}\'
    """

    # SQL to check if schema exists
    check_schema_sql = f"""
    SELECT 1 FROM information_schema.schemata WHERE schema_name = \'{pg_schema.upper()}\'
    """

    # SQL to create schema if it doesn't exist
    create_schema_sql = f"""
    CREATE SCHEMA IF NOT EXISTS \'{pg_schema.upper()}\'
    """

    # SQL to create the function
    create_function_sql = (f"""
    CREATE OR REPLACE FUNCTION \"{pg_schema}\".elocation_edge_link_level(func_class bigint) 
    RETURNS bigint 
    AS $body$
    BEGIN
    RETURN floor((8 - func_class) / 3);
    END;
    $body$
    LANGUAGE PLPGSQL
    SECURITY DEFINER
    IMMUTABLE
    """)

    
    try:
        pg_cursor.execute(check_role_sql)
        role_exists = pg_cursor.fetchone() is not None
        # if not role_exists:
            # print("Warning: Role 'HERE_SF' does not exist. Creating function might fail.")
        
        # Check if schema exists, create if it doesn't
        pg_cursor.execute(check_schema_sql)
        schema_exists = pg_cursor.fetchone() is not None
        if not schema_exists:
            # print("Schema 'HERE_SF' does not exist. Creating it now.")
            pg_cursor.execute(create_schema_sql)
        
        # Execute the function creation SQL
        pg_cursor.execute(create_function_sql)
        
        # Attempt to change the owner (this might fail if role doesn't exist)
        try:
            pg_cursor.execute('ALTER FUNCTION \"HERE_SF\".elocation_edge_link_level(bigint) OWNER TO \"HERE_SF\"')
        except psycopg2.Error as e:
            logging.info(f"Warning: Couldn't change function owner: {e}")
        
        # Commit the transaction
        
        logging.info("Function ELOCATION_EDGE_LINK_LEVEL created successfully.")


    except psycopg2.Error as e:
        logging.info(f"Failed to create ELOCATION_EDGE_LINK_LEVEL function: {str(e)}")
        # st.error(f"Failed to create ELOCATION_EDGE_LINK_LEVEL function: {str(e)}")

def create_soundex_function(pg_cursor, pg_schema):
    logging.info("Performing migration of Soundex function")
    function_sql = sql.SQL("""
    CREATE OR REPLACE FUNCTION {schema}.\"soundex\"(text) RETURNS text AS $$
    DECLARE
        str TEXT := upper($1);
        c TEXT;
        last_code TEXT := '';
        curr_code TEXT := '';
        soundex_code TEXT := '';
    BEGIN
        IF str IS NULL OR str = '' THEN
            RETURN NULL;
        END IF;

        soundex_code := substr(str, 1, 1);  -- first letter

        FOR i IN 2..length(str) LOOP
            c := substr(str, i, 1);
            CASE c
                WHEN 'B', 'F', 'P', 'V' THEN curr_code := '1';
                WHEN 'C', 'G', 'J', 'K', 'Q', 'S', 'X', 'Z' THEN curr_code := '2';
                WHEN 'D', 'T' THEN curr_code := '3';
                WHEN 'L' THEN curr_code := '4';
                WHEN 'M', 'N' THEN curr_code := '5';
                WHEN 'R' THEN curr_code := '6';
                ELSE curr_code := '';
            END CASE;

            IF curr_code <> '' AND curr_code <> last_code THEN
                soundex_code := soundex_code || curr_code;
            END IF;

            last_code := curr_code;

            IF length(soundex_code) = 4 THEN
                EXIT;
            END IF;
        END LOOP;

        RETURN soundex_code || repeat('0', 4 - length(soundex_code));
    END;
    $$ LANGUAGE plpgsql IMMUTABLE;
    """).format(schema=sql.Identifier(pg_schema))
    
    pg_cursor.execute(function_sql)
    
    
    
def get_user_indexes(cursor, username):
    
    cursor.execute("""
        SELECT UPPER(i.index_name), UPPER(i.table_name), 
               LISTAGG(UPPER(c.column_name), ', ') WITHIN GROUP (ORDER BY c.column_position) as column_info, 
               UPPER(i.uniqueness)
        FROM all_indexes i
        JOIN all_ind_columns c ON i.index_name = c.index_name AND i.table_name = c.table_name AND i.table_owner = c.table_owner
        WHERE i.table_owner = :username
        GROUP BY i.index_name, i.table_name, i.uniqueness
        ORDER BY i.index_name
    """, username=username.upper())
    return cursor.fetchall()

def save_indexes_to_config(indexes, filename='default_indexes.json'):
    with open(filename, 'w') as file:
        json.dump(indexes, file, indent=4)

def load_indexes_from_config(filename='default_indexes.json'):
    with open(filename, 'r') as file:
        return json.load(file)

def check_index_exists(pg_cursor, pg_schema, index_name, table_name, column_name):
    try:
        pg_cursor.execute(
            sql.SQL('SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE schemaname = %s AND indexname = %s AND tablename = %s)'),
            (pg_schema, index_name, table_name)
        )
        return pg_cursor.fetchone()[0]
    except psycopg2.Error as e:
        logging.info(f'Error checking if index "{pg_schema}"."{index_name}" exists: {str(e)}')
        st.write(f'Error checking if index "{pg_schema}"."{index_name}" exists: {str(e)}')
        error_messages.append(f'Error checking if index "{pg_schema}"."{index_name}" exists: {str(e)}')
        return False

def check_table_exists(pg_cursor, pg_schema, table_name):
    try:
        pg_cursor.execute(
            sql.SQL('SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s)'),
            (pg_schema, table_name)
        )
        return pg_cursor.fetchone()[0]
    except psycopg2.Error as e:
        st.write(f'Error checking if table "{pg_schema}"."{table_name}" exists: {str(e)}')
        logging.info(f'Error checking if table "{pg_schema}"."{table_name}" exists: {str(e)}')
        error_messages.append(f'Error checking if table "{pg_schema}"."{table_name}" exists: {str(e)}')
        return False

def migrate_indexes(oracle_indexes, pg_conn, pg_schema, expander):
    success_count = 0
    failure_count = 0
    skipped_indexes = []
    try:
        with pg_conn.cursor() as pg_cursor:
            create_elocation_edge_link_level_function(pg_cursor, pg_schema)
            create_soundex_function(pg_cursor, pg_schema)
            pg_conn.commit()
    except psycopg2.Error as e:
        expander.write(f"Failed to set up necessary functions: {str(e)}")
        pg_conn.rollback()
    
    for oracle_index in oracle_indexes:
        index_name = oracle_index[0]  # Assuming index_name is the first element
        try:
            with pg_conn.cursor() as pg_cursor:
                if migrate_index(oracle_index, pg_cursor, pg_schema, expander):
                    success_count += 1
                    pg_conn.commit()
                else:
                    failure_count += 1
                    skipped_indexes.append(index_name)
                    pg_conn.rollback()
        except psycopg2.Error as e:
            failure_count += 1
            skipped_indexes.append(index_name)
            pg_conn.rollback()
    return success_count, failure_count, skipped_indexes
    

def migrate_index(oracle_index, pg_cursor, pg_schema, expander):
    
    index_name, table_name, column_info, uniqueness = oracle_index
    unique_clause = "UNIQUE" if uniqueness == "UNIQUE" else ""
    
    try:
        if not check_table_exists(pg_cursor, pg_schema, table_name):
            logging.info(f'Table "{pg_schema}"."{table_name}" does not exist in PostgreSQL. Skipping index "{index_name}".')
            expander.write(f'Table "{pg_schema}"."{table_name}" does not exist in PostgreSQL. Skipping index "{index_name}".')
            return False
        
        if check_index_exists(pg_cursor, pg_schema, index_name, table_name, column_info):
            logging.info(f'Index "{index_name}" already exists in the database. Skipping.')
            expander.write(f'Index "{index_name}" already exists in the database. Skipping.')
            return True
        
        # Parse the column_info to handle complex cases
        if 'USING gist' in column_info.lower():
            method = 'USING gist'
            column_info = re.sub(r'\s*USING\s+gist.*', '', column_info, flags=re.IGNORECASE)
        else:
            method = ''
        
        # Handle functions in column definitions
        columns = [col.strip() for col in column_info.split(',')]
        formatted_columns = []
        # print(columns,'\n\n')
        for col in columns:
            # print(col)
            if 'elocation_edge_link_level(' in col.lower():
                # Replace with the correct schema-qualified function call
                col = col.replace('elocation_edge_link_level(', f'\"{pg_schema}\".\"ELOCATION_EDGE_LINK_LEVEL\"(')
                formatted_columns.append(sql.SQL(col))
            elif '(' in col and ')' in col:
                # This is likely a different function call
                formatted_columns.append(sql.SQL(col))
            else:
                formatted_columns.append(sql.Identifier(col))
        
        column_sql = sql.SQL(', ').join(formatted_columns)
        
        create_index_sql = sql.SQL('CREATE {unique} INDEX {index_name} ON {schema}.{table} {method} ({columns})').format(
            unique=sql.SQL(unique_clause),
            index_name=sql.Identifier(index_name),
            schema=sql.Identifier(pg_schema),
            table=sql.Identifier(table_name),
            method=sql.SQL(method),
            columns=column_sql
        )
        # print(f'CREATE {unique_clause} INDEX {index_name} ON {pg_schema}.{table_name} {method} ({column_sql})')
        
        pg_cursor.execute(create_index_sql)
        expander.write(f'Successfully migrated index: \"{pg_schema}\".\"{index_name}\"')
        logging.info(f'Successfully migrated index: \"{pg_schema}\".\"{index_name}\"')
        return True
    except psycopg2.Error as e:
        expander.write(f'Failed to migrate index \"{pg_schema}\".\"{index_name}\": {str(e)}')
        logging.info(f'Failed to migrate index \"{pg_schema}\".\"{index_name}\": {str(e)}')
        return False




def fetch_solutions_for_errors(error_messages):
    try:
        solution_prompts = []

        for error_message in error_messages:
            prompt = f"Error occurred during migration:\n\n{error_message}"
            GOOGLE_API_KEY = "AIzaSyANXuJrBgTaReoX5yU040oSOSMzFAZNEGI"
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
            response = llm.invoke(prompt)
            solution_prompts.append(response.content.strip())

        return solution_prompts
    except:
        st.error("LLM access cannot be configured. Contact Promptora AI helpline:\nkeerthi@aipromptora.com\naparna@aipromptora.com\nganesh.gadhave@aipromptora.com")

def create_schema(pg_cursor, pg_schema):
    try:
        pg_cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(pg_schema)))
        st.write(f"Created schema: {pg_schema}")
    except psycopg2.Error as e:
        st.error(f"Failed to create schema {pg_schema}: {str(e)}")
        error_messages.append(f"Failed to create schema {pg_schema}: {str(e)}")
        exit(1)
        
        
def extract_functions(schema_name,oracle_conn):
    schema_name = schema_name.upper()
    with oracle_conn as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT object_name
                FROM all_procedures
                WHERE owner = :schema_name
                AND object_type = 'FUNCTION'
            """, schema_name=schema_name)
            functions = cur.fetchall()
            functions_dict = {}
            for func in functions:
                func_name = func[0]
                cur.execute("""
                    SELECT text
                    FROM all_source
                    WHERE owner = :schema_name
                    AND name = :object_name
                    AND type = 'FUNCTION'
                    ORDER BY line
                """, schema_name=schema_name, object_name=func_name)
                text_lines = [row[0] for row in cur.fetchall()]
                ddl_text = ''.join(text_lines)
                functions_dict[func_name] = ddl_text
    return functions_dict
 
def get_llm_suggestion(functions_dict):
    try:
        if not functions_dict:
            return "No functions found to migrate."

        prompt = "I'm migrating an Oracle database to PostgreSQL. Here are the functions I need to migrate:\n\n"
        for name, ddl in functions_dict.items():
            prompt += f"FUNCTION '{name}':\n{ddl}\n\n"
        prompt += "\nProvide the PostgreSQL equivalent for these functions."

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key="AIzaSyANXuJrBgTaReoX5yU040oSOSMzFAZNEGI")
        response = llm.invoke(prompt)

        clean_response = response.content.strip()

        start_marker = "```sql"
        end_marker = "```"

        start_index = clean_response.find(start_marker)
        end_index = clean_response.find(end_marker, start_index + len(start_marker))

        if start_index != -1 and end_index != -1:
            clean_response = clean_response[start_index + len(start_marker):end_index].strip()
        else:
            clean_response = clean_response.strip()

        clean_response = '\n'.join(line for line in clean_response.split('\n') if not line.strip().startswith('--'))

        return clean_response
    except:
        st.error("LLM access cannot be configured. Contact Promptora AI helpline:\nkeerthi@aipromptora.com\naparna@aipromptora.com\nganesh.gadhave@aipromptora.com")


def format_function_ddl(schema_name, function_name, function_ddl):
    schema_name = schema_name.upper()
    function_name = function_name.upper()

    if 'CREATE OR REPLACE FUNCTION' in function_ddl:
        function_def = function_ddl.split('CREATE OR REPLACE FUNCTION', 1)[1].strip()

        if '(' in function_def:
            function_name_part = function_def.split('(', 1)[0].strip()
            params_part = function_def.split('(', 1)[1].strip()
        else:
            function_name_part = function_def
            params_part = ''

        function_ddl = f'CREATE OR REPLACE FUNCTION "{schema_name}"."{function_name}"({params_part}'
    else:
        raise ValueError("Invalid function definition.")

    function_ddl = function_ddl.replace('`', '')

    return function_ddl



def migrate_function(schema_name, pg_connection,function_name, function_ddl):
    formatted_ddl = format_function_ddl(schema_name, function_name, function_ddl)

    try:
        with pg_connection as conn:
            with conn.cursor() as cur:
                cur.execute(formatted_ddl)
                conn.commit()
                st.write(f"FUNCTION '{function_name}' migrated successfully to schema '{schema_name}'.")
    except Exception as e:
        error_message = f"Error migrating FUNCTION '{function_name}': {str(e)}"
        st.write(error_message)
        function_errors.append(error_message)
 




# def migrate_function(schema_name, function_name, function_ddl,pg_connection, expander):
#     formatted_ddl = format_function_ddl(schema_name, function_name, function_ddl) 
#     try:
#         with pg_connection as conn:
#             with conn.cursor() as cur:
#                 cur.execute(formatted_ddl)
#                 conn.commit()
#                 expander.write(f"Function '{function_name}' migrated successfully to schema '{schema_name}'.")
#     except Exception as e:
#         st.write(f"Error migrating FUNCTION '{function_name}': {str(e)}")



# def format_function_ddl(schema_name, function_name, function_ddl):
#     schema_name = schema_name.upper()
#     function_name = function_name.lower()   
#     if 'CREATE OR REPLACE FUNCTION' in function_ddl:
#         function_def = function_ddl.split('CREATE OR REPLACE FUNCTION', 1)[1].strip()
#         if '(' in function_def:
#             function_name_part = function_def.split('(', 1)[0].strip()
#             params_part = function_def.split('(', 1)[1].strip()
#         else:
#             function_name_part = function_def
#             params_part = ''
#         function_ddl = f'CREATE OR REPLACE FUNCTION "{schema_name}"."{function_name}"({params_part}'
#     else:
#         raise ValueError("Invalid function definition.")
#     function_ddl = function_ddl.replace('`', '')
#     return function_ddl
 

 
 
def create_schema_if_not_exists_fun(schema_name,pg_connection):
    schema_name = schema_name.upper()
    create_schema_sql = f'CREATE SCHEMA IF NOT EXISTS "{schema_name}";'

    try:
        with pg_connection as conn:
            with conn.cursor() as cur:
                cur.execute(create_schema_sql)
                conn.commit()
                st.write(f"Schema '{schema_name}' is created or already exists.")
    except Exception as e:
        error_message = f"Error creating schema '{schema_name}': {str(e)}"
        st.write(error_message)
        error_messages.append(error_message)

def migrate_schema_objects_fun(schema_name,oracle_conn,pg_connection):
    st.write(f"Extracting functions for schema '{schema_name}'...")
    functions_dict = extract_functions(schema_name,oracle_conn)

    if not functions_dict:
        st.write("No functions found to migrate.")
        return

    create_schema_if_not_exists_fun(schema_name,pg_connection)

    st.write(f"Migrating {len(functions_dict)} functions to PostgreSQL...")
    for func_name, ddl in functions_dict.items():
        st.write(f"Migrating FUNCTION '{func_name}'...")

        # Create a dictionary in the correct format
        func_dict = {func_name: ddl}

        llm_suggestion = get_llm_suggestion(func_dict)
        migrate_function(schema_name,pg_connection, func_name, llm_suggestion)
        

def fetch_solutions_for_errors(error_messages):
    try:
        solution_prompts = []
        for error_message in error_messages:
            prompt = f"Error occurred during migration:\n\n{error_message}"
            GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"  # Replace with your actual Google API Key
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
            response = llm.invoke(prompt)
            solution_prompts.append(response.content.strip())
        return solution_prompts
    except Exception as e:
        st.error(f"LLM access cannot be configured. Contact Promptora AI helpline: keerthi@aipromptora.com, aparna@aipromptora.com, ganesh.gadhave@aipromptora.com\nError: {str(e)}")
        return []



###############################################   PROCEDURES   ###############################################


def extract_procedures(schema_name,oracle_conn):
    schema_name = schema_name.upper()
    with oracle_conn as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT object_name
                FROM all_procedures
                WHERE owner = :schema_name
                AND object_type = 'PROCEDURE'
            """, schema_name=schema_name)
            procedures = cur.fetchall()
            procedures_dict = {}
            for proc in procedures:
                proc_name = proc[0]
                cur.execute("""
                    SELECT text
                    FROM all_source
                    WHERE owner = :schema_name
                    AND name = :object_name
                    AND type = 'PROCEDURE'
                    ORDER BY line
                """, schema_name=schema_name, object_name=proc_name)
                text_lines = [row[0] for row in cur.fetchall()]
                ddl_text = ''.join(text_lines)
                procedures_dict[proc_name] = ddl_text
    return procedures_dict

def format_procedure_ddl(schema_name, procedure_name, procedure_ddl):
    schema_name = schema_name.upper()
    procedure_name = procedure_name.upper()
    try:
        # Extract the procedure body
        body_match = re.search(r'IS\s+BEGIN(.+?)END', procedure_ddl, re.DOTALL)
        if not body_match:
            raise ValueError("Cannot extract procedure body")
        body_part = body_match.group(1).strip()
        # Convert Oracle-specific functions to PostgreSQL equivalents
        body_part = body_part.replace("TO_CHAR (SYSDATE, 'HH24:MI')", "TO_CHAR(CURRENT_TIME, 'HH24:MI')")
        body_part = body_part.replace("TO_CHAR (SYSDATE, 'DY')", "TRIM(TO_CHAR(CURRENT_DATE, 'Day'))")
        # Convert RAISE_APPLICATION_ERROR to RAISE EXCEPTION
        body_part = re.sub(
            r"RAISE_APPLICATION_ERROR\s*\(\s*-?\d+\s*,\s*'(.+?)'\s*\)\s*;",
            r"RAISE EXCEPTION '\1';",
            body_part
        )
        # Ensure proper formatting of IF...THEN...END IF blocks
        def format_if_blocks(text):
            result = []
            lines = text.splitlines()
            indent = 0
            in_if_block = False
            if_conditions = []
            for line in lines:
                stripped_line = line.strip()
                if stripped_line.startswith('IF'):
                    if in_if_block:
                        # Close previous IF block if it's still open
                        result.append(' ' * indent + 'END IF;')
                    if_conditions = [stripped_line]
                    result.append(' ' * indent + stripped_line + ' THEN')
                    indent += 4
                    in_if_block = True
                elif stripped_line.startswith('ELSE'):
                    result.append(' ' * (indent - 4) + stripped_line)
                elif stripped_line.startswith('ELSIF'):
                    result.append(' ' * (indent - 4) + stripped_line + ' THEN')
                elif stripped_line.startswith('END IF'):
                    indent -= 4
                    result.append(' ' * indent + stripped_line)
                    in_if_block = False
                else:
                    if in_if_block:
                        if stripped_line.startswith('OR') or stripped_line.startswith('AND'):
                            # Add logical operators to the last IF condition
                            if_conditions.append(stripped_line)
                        else:
                            result.append(' ' * indent + stripped_line)
                    else:
                        result.append(stripped_line)
            # Close any open IF blocks
            if in_if_block:
                result.append('END IF;')
            return '\n'.join(result)
        body_part = format_if_blocks(body_part)
        # Remove any unnecessary trailing semicolons before END;
        body_part = re.sub(r';\s*END\s*$', '\nEND;', body_part)
        # Extract parameters (if any)
        params_match = re.search(r'\((.*?)\)', procedure_ddl.split('IS')[0])
        params = params_match.group(1) if params_match else ''
        # Convert Oracle parameter types to PostgreSQL types
        params = re.sub(r'(\w+)%TYPE', r'\1', params)
        # Add default types for parameters if not specified
        if params:
            param_list = [p.strip() for p in params.split(',')]
            typed_params = []
            for param in param_list:
                if ' ' not in param:
                    param += ' VARCHAR'  # Default type
                typed_params.append(param)
            params = ', '.join(typed_params)
        # Construct PostgreSQL procedure DDL
        formatted_ddl = f'''
CREATE OR REPLACE PROCEDURE "{schema_name}"."{procedure_name}"({params})
LANGUAGE plpgsql
AS $$
BEGIN
{body_part}
END;
$$;
'''
    except Exception as e:
        raise ValueError(f"Error formatting procedure DDL: {str(e)}")
    return formatted_ddl

def migrate_procedure(pg_connection,schema_name, procedure_name, procedure_ddl):
    try:
        formatted_ddl = format_procedure_ddl(schema_name, procedure_name, procedure_ddl)
        with pg_connection as conn:
            with conn.cursor() as cur:
                cur.execute(formatted_ddl)
                conn.commit()
                st.write(f"PROCEDURE '{procedure_name}' migrated successfully to schema '{schema_name}'.")
                st.write("Migrated DDL:")
                st.code(formatted_ddl, language="sql")
    except ValueError as ve:
        error_message = f"Error formatting PROCEDURE '{procedure_name}': {str(ve)}"
        st.write(error_message)
        error_messages.append(error_message)
    except psycopg2.Error as pe:
        error_message = f"Error executing PROCEDURE '{procedure_name}': {str(pe)}"
        st.write(error_message)
        error_messages.append(error_message)
        st.write("Attempted DDL:")
        st.code(formatted_ddl, language="sql")
    except Exception as e:
        error_message = f"Unexpected error migrating PROCEDURE '{procedure_name}': {str(e)}"
        st.write(error_message)
        error_messages.append(error_message)

def create_schema_if_not_exists(pg_connection,schema_name):
    schema_name = schema_name.upper()
    create_schema_sql = f'CREATE SCHEMA IF NOT EXISTS "{schema_name}";'
    try:
        with pg_connection as conn:
            with conn.cursor() as cur:
                cur.execute(create_schema_sql)
                conn.commit()
                st.write(f"Schema '{schema_name}' is created or already exists.")
    except Exception as e:
        error_message = f"Error creating schema '{schema_name}': {str(e)}"
        st.write(error_message)
        error_messages.append(error_message)

def migrate_schema_objects(schema_name,oracle_conn,pg_connection):
    st.write(f"Extracting procedures for schema '{schema_name}'...")
    procedures_dict = extract_procedures(schema_name,oracle_conn)
    if not procedures_dict:
        st.write("No procedures found to migrate.")
        return
    create_schema_if_not_exists(pg_connection,schema_name)
    st.write(f"Migrating {len(procedures_dict)} procedures to PostgreSQL...")
    for proc_name, ddl in procedures_dict.items():
        st.write(f"Migrating PROCEDURE '{proc_name}'...")
        migrate_procedure(pg_connection,schema_name, proc_name, ddl)

###############################################   SYNONYMS AND SEQUENCES AND DBLINKS   ###############################################




def seq_syn_migrate(oracle_conn,pg_conn,schema, object_type):
    
    oracle_cursor=oracle_conn.cursor()
    pg_cursor=pg_conn.cursor()
    oracle_cursor.execute(f"SELECT object_name FROM all_objects WHERE object_type = :type AND owner = :schema",
                            type=object_type.upper(), schema=schema.upper())
    object_names = oracle_cursor.fetchall()
    for (object_name,) in object_names:
        st.write(f"Processing {object_type}: {object_name}")
        try:
            if object_type == 'synonym':
                migrate_synonym(oracle_cursor, pg_cursor, object_name, schema)
            elif object_type == 'sequence':
                migrate_sequence(oracle_cursor, pg_cursor, object_name, schema)
            elif object_type == 'dblink':
                migrate_dblink(oracle_cursor, pg_cursor, object_name, schema)
            pg_conn.commit()
        except Exception as e:
            st.write(f"{object_type} {object_name}: {str(e)}")
            pg_conn.rollback()


# Function to migrate a sequence
def migrate_sequence(oracle_cursor, pg_cursor, sequence_name, schema):
    st.write(f"Migrating the sequence named: {sequence_name}")
    oracle_cursor.execute("SELECT last_number, min_value, max_value, increment_by FROM all_sequences WHERE sequence_name = :name AND sequence_owner = :schema",
                          name=sequence_name, schema=schema)
    seq_info = oracle_cursor.fetchone()
    if seq_info:
        last_number, min_value, max_value, increment_by = seq_info
        # Adjust values to fit PostgreSQL limits
        pg_max_value = min(max_value, 9223372036854775807)
        pg_last_number = min(last_number, pg_max_value)
        pg_cursor.execute(f"""CREATE SEQUENCE "{schema}"."{sequence_name}" 
                              START WITH {pg_last_number} 
                              MINVALUE {min_value} 
                              MAXVALUE {pg_max_value} 
                              INCREMENT BY {increment_by}""")
        st.write(f"Migrated sequence: {sequence_name}")

# Function to migrate a synonym
def migrate_synonym(oracle_cursor, pg_cursor, synonym_name, schema):
    st.write(f"Migrating the synonym named: {synonym_name}")
    oracle_cursor.execute("SELECT table_owner, table_name FROM all_synonyms WHERE synonym_name = :name AND owner = :schema", 
                          name=synonym_name, schema=schema)
    synonym_info = oracle_cursor.fetchone()
    if synonym_info:
        table_owner, table_name = synonym_info
        pg_cursor.execute(f"""CREATE VIEW "{schema}"."{synonym_name}" AS SELECT * FROM \"{table_owner}\".\"{table_name}\"""")
        st.write(f"Migrated synonym: {synonym_name}")

def migrate_dblink(oracle_cursor, pg_cursor, dblink_name, schema):
    st.write(f"Migrating the dblink named: {dblink_name}")
    oracle_cursor.execute("SELECT host, username, db_link FROM all_db_links WHERE db_link = :name AND owner = :schema",
                          name=dblink_name, schema=schema)
    dblink_info = oracle_cursor.fetchone()
    if dblink_info:
        host, username, db_link = dblink_info
        st.write(f"Warning: DB Link '{dblink_name}' cannot be directly migrated. Consider using Foreign Data Wrappers in PostgreSQL.")



###############################################   VALIDATION   ###############################################



def get_oracle_object_details(schema_name, oracle_conn_str):
    connection = cx_Oracle.connect(oracle_conn_str)
    cursor = connection.cursor()
    query = f"""
    SELECT object_type,
        COUNT(*) AS object_count
    FROM all_objects a
    WHERE owner = '{schema_name}'
    GROUP BY object_type
    ORDER BY object_type
    """
    cursor.execute(query)
    # print("Got oracle data")
    oracle_objects = cursor.fetchall()
    cursor.close()
    connection.close()
    return oracle_objects
 
def get_postgres_object_details(schema_name, conn_str):
    connection = psycopg2.connect(conn_str)
    cursor = connection.cursor()
 
    query = f"""
    SELECT CASE
            WHEN c.relkind = 'r' THEN 'TABLE'
            WHEN c.relkind = 'v' THEN 'VIEW'
            WHEN c.relkind = 'f' THEN 'FOREIGN TABLE'
            WHEN c.relkind = 'S' THEN 'SEQUENCE'
            WHEN c.relkind = 'I' THEN 'INDEX'
            WHEN c.relkind = 'c' THEN 'TYPE'
            WHEN c.relkind = 'p' THEN 'PARTITIONED TABLE'
        END AS object_type,
        COUNT(*) AS object_count
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = '{schema_name}'
    GROUP BY object_type
    ORDER BY object_type
    """ 
    cursor.execute(query)
    # print("Got Postgres data")
    postgres_objects = cursor.fetchall()
    cursor.close()
    connection.close() 
    return postgres_objects


def llm_validation(oracle_objects, postgres_objects):
    try:
        prompt = f"""Compare the Oracle and PostgreSQL schemas and generate a concise report:
        Oracle Objects:
        {oracle_objects}
        PostgreSQL Objects:
        {postgres_objects}
        Please note that there is going to be an extra table in postgres because of the installation of extension postgis. Same goes for view, there are two views under the name geometry_columns and geography_columns
        Please provide a summary of the differences and similarities between the two schemas,
        including object types, counts, and any notable discrepancies in structure or data types."""
        GOOGLE_API_KEY = "AIzaSyANXuJrBgTaReoX5yU040oSOSMzFAZNEGI"
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
        # print("LLM Configured")
        response = llm.invoke(prompt)
        with st.expander("Insights(_Generated by language model_):"):
            st.write(f"```{response.content.strip()}")
    except:
        st.error("LLM access cannot be configured. Contact Promptora AI helpline:\nkeerthi@aipromptora.com\naparna@aipromptora.com\nganesh.gadhave@aipromptora.com")
 
def validating(schema,oracle_conn_str, postgres_conn_str):
    st.title("Database Schema Comparison")
    oracle_objects = get_oracle_object_details(schema, oracle_conn_str)
    postgres_objects = get_postgres_object_details(schema, postgres_conn_str)
    llm_validation(oracle_objects, postgres_objects)



    
###############################################   VIEWS   ###############################################




def get_oracle_views_in_schema(oracle_cur, schema_name):
    try:
        oracle_cur.execute(f"""SELECT *
                           FROM all_views
                           WHERE owner = :schema_name
                           ORDER BY view_name""", schema_name=schema_name)
       
        column_names = [d[0] for d in oracle_cur.description]
        view_name_column = next((col for col in column_names if 'NAME' in col.upper()), None)
        view_name_index = column_names.index(view_name_column)
        return [row[view_name_index] for row in oracle_cur.fetchall()]
    except cx_Oracle.Error as error:
        st.error(f"Error fetching views from Oracle: {error}")
        return []
    
def get_oracle_view(oracle_cur, schema_name, view_name):
    try:
        oracle_cur.execute(f"""SELECT text_vc
                           FROM all_views
                           WHERE owner = :schema_name AND view_name = :view_name""",
                           schema_name=schema_name, view_name=view_name)
        result = oracle_cur.fetchone()
        if result:
            view_text = result[0].read() if isinstance(result[0], cx_Oracle.LOB) else result[0]
            # Convert table names to uppercase
            view_text = re.sub(r'\b([a-z_]\w*)\b', lambda m: m.group(1).upper(), view_text, flags=re.IGNORECASE)
            return view_text
        return None
    except cx_Oracle.Error as error:
        st.error(f"Error fetching view {view_name} from Oracle: {error}")
        return None
    
def check_postgres_table_exists(pg_cur, table_name, schema_name):
    try:
        pg_cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = :schema_name
            AND upper(table_name) = upper(%s)
        """, (table_name,), schema_name=schema_name)
       
        result = pg_cur.fetchone()
        return result is not None
    except psycopg2.Error as error:
        st.error(f"Error checking if dependent table exists in PostgreSQL: {error}")
        return False

def migrate_view(view_name, view_text, pg_cur, pg_conn, schema_name):
    pg_view_name = f'"{view_name.upper()}"'
    pg_view_text = convert_oracle_to_postgres(view_text,schema_name)
    schema_name = f'"{schema_name.upper()}"'
    # print('\n\n',pg_view_text,'\n\n')
    full_query = f'CREATE OR REPLACE VIEW {schema_name}.{pg_view_name} AS {pg_view_text}'
    try:
        pg_cur.execute(full_query)
        st.write(f"Successfully migrated view: {view_name}")
        pg_conn.commit()
        return True
    except psycopg2.Error as error:
        st.write(f"Failed to migrate view {view_name}: {str(error)}")
        # print("Error details:")
        # print(error.pgerror)
        # print("Query that caused the error:")
        # print(full_query)
        return False
    
def convert_oracle_to_postgres(oracle_view_text, schema_name):
    conversions = {
        'NVL\((.*?),(.*?)\)': 'COALESCE(\g<1>,\g<2>)',
        'TRUNC\((.*?)\)': 'DATE_TRUNC(\'day\', \g<1>)',
        'SYSDATE': 'CURRENT_DATE',
        'DECODE\((.*?)\)': lambda m: convert_decode_to_case(m.group(1)),
        '([A-Za-z0-9_]+)\.NEXTVAL': "nextval('\g<1>_seq')",
    }
    for oracle_syntax, pg_syntax in conversions.items():
        if callable(pg_syntax):
            oracle_view_text = re.sub(oracle_syntax, pg_syntax, oracle_view_text)
        else:
            oracle_view_text = re.sub(oracle_syntax, pg_syntax, oracle_view_text, flags=re.IGNORECASE)
    oracle_view_text = re.sub(r'\s*WITH READ ONLY\s*$', '', oracle_view_text, flags=re.IGNORECASE)
    parts = re.split(r'\bFROM\b', oracle_view_text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        select_part, from_part = parts
        select_part = select_part.replace('SELECT', '', 1).strip()  # Remove the first SELECT
    else:
        select_part = oracle_view_text.replace('SELECT', '', 1).strip()
        from_part = ""

    select_part = re.sub(r'(\b\w+\b)', lambda m: f'"{m.group(1).upper()}"', select_part)
    if from_part:
        schema_name=f'"{schema_name.upper()}"'
        from_part = re.sub(r'\b([a-z_]\w*)\b', lambda m: f'"{m.group(1).upper()}"', from_part, flags=re.IGNORECASE)
        from_part=from_part.strip()
        oracle_view_text = f"SELECT {select_part} FROM {schema_name}.{from_part}"
    else:
        oracle_view_text = f"SELECT {select_part}"
    return oracle_view_text

def convert_decode_to_case(decode_args):
    args = [arg.strip() for arg in decode_args.split(',')]
    column = args[0]
    pairs = list(zip(args[1::2], args[2::2]))
    case_stmt = f"CASE {column}"
    for when, then in pairs[:-1]:
        case_stmt += f" WHEN {when} THEN {then}"
    case_stmt += f" ELSE {pairs[-1][1]} END"
    return case_stmt
 

def extract_tables_from_view(view_text):
    parsed = sqlparse.parse(view_text)[0]
    tables = set()
   
    def extract_from_token(token):
        if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
            idx = token.parent.token_index(token)
            next_token = token.parent.token_next(idx)[1]
            if isinstance(next_token, sqlparse.sql.Identifier):
                tables.add(next_token.get_real_name().upper())
            elif isinstance(next_token, sqlparse.sql.IdentifierList):
                for identifier in next_token.get_identifiers():
                    tables.add(identifier.get_real_name().upper())
 
    for token in parsed.flatten():
        extract_from_token(token)
 
    return list(tables)

def view_migration(oracle_conn,pg_conn,schema_name):
   
    oracle_cur = oracle_conn.cursor()
  
    pg_cur = pg_conn.cursor()

    views_in_schema = get_oracle_views_in_schema(oracle_cur, schema_name)
    st.write(f"Available views in schema :blue['{schema_name}']:")
    for view in views_in_schema:
        st.write(view)
    total_views = len(views_in_schema)
    migrated_views = 0
    skipped_views = 0
    for view_name in views_in_schema:
        view_text = get_oracle_view(oracle_cur, schema_name, view_name)
        if view_text is None:
            st.info(f"Skipping view {schema_name}.{view_name} due to missing or empty view text")
            logging.info(f"Skipping view {schema_name}.{view_name} due to missing or empty view text")
            skipped_views += 1
            continue 
        
        st.write(f"\nProcessing view: {schema_name}.{view_name}")
        pg_view_text = convert_oracle_to_postgres(view_text, schema_name)
        st.write(f"Converted SQL for view '{view_name}':")
        st.write(f':orange[{pg_view_text}]')
        tables = extract_tables_from_view(view_text)
        st.write(f"Tables referenced in the view: {tables}")
        if migrate_view(view_name, view_text, pg_cur, pg_conn, schema_name):
            migrated_views += 1
            logging.info(f"Migrated view: {view_name} successfully!")
        else:
            skipped_views += 1
        st.divider()

    st.write("\nMigrated Views Successfully!")
    st.divider
    st.write(f"Total views processed: {total_views}")
    st.write(f"Views migrated: {migrated_views}")
    st.write(f"Views skipped: {skipped_views}")    




################################################## Triggers ################################



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
        st.write(f"Oracle Error: {e}")
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
            st.write("Warning: CONNECT BY clause detected. This may require manual conversion to WITH RECURSIVE.")

        return trigger_body
    except Exception as e:
        st.write(f"Error in convert_trigger_body: {e}")
        return trigger_body

def get_llm_suggestion_trig(error_message, trigger_name, trigger_body):
    try:
        prompt = f"""
I'm migrating an Oracle trigger to PostgreSQL and encountered the following error:
Error in trigger '{trigger_name}': {error_message}

Here's the original trigger body:
{trigger_body}

How should I modify this trigger to work in PostgreSQL? Please provide a step-by-step approach and, if possible, a corrected version of the trigger.
"""

        GOOGLE_API_KEY = "AIzaSyANXuJrBgTaReoX5yU040oSOSMzFAZNEGI"
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
        response = llm.invoke(prompt)

        return response.content
    except Exception as e:
        return f"LLM access error: {str(e)}. Contact Promptora AI helpline: keerthi@aipromptora.com, aparna@aipromptora.com, ganesh.gadhave@aipromptora.com"

def migrate_trigger_to_postgres(pg_conn, schema_name, trigger_name, triggering_event, trigger_type, trigger_body, table_name):
    cursor = pg_conn.cursor()

    try:
        # Set the search path to the schema in PostgreSQL, preserving capitalization
        cursor.execute(f'SET search_path TO "{schema_name}";')

        # Check if the schema exists
        cursor.execute(f"SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = %s)", (schema_name,))
        schema_exists = cursor.fetchone()[0]

        if not schema_exists:
            st.write(f"Schema '{schema_name}' does not exist in PostgreSQL. Creating it now.")
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
            st.write(f"Trigger {pg_trigger_name} already exists in PostgreSQL. Skipping.")
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
        st.write(f"Trigger {trigger_name} migrated successfully.")

    except psycopg2.Error as e:
        pg_conn.rollback()  # Rollback transaction if there's an error
        error_message = str(e)
        st.write(f"Error executing trigger {trigger_name}: {error_message}")
        
        # Get LLM suggestion
        suggestion = get_llm_suggestion_trig(error_message, trigger_name, trigger_body)
        st.write(f"\nLLM Suggestion for {trigger_name}:\n{suggestion}")

    finally:
        cursor.close()

def migrate_triggers(oracle_conn_str,postgres_conn_str, schema_name):
    oracle_conn = None
    pg_conn = None
    try:
        # Connect to Oracle and PostgreSQL
        oracle_conn = cx_Oracle.connect(oracle_conn_str)
        pg_conn = psycopg2.connect(postgres_conn_str)

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
                st.write(f"Error migrating trigger {trigger[0]}: {e}")
                # Get LLM suggestion for general migration errors
                suggestion = get_llm_suggestion_trig(str(e), trigger[0], trigger[3])
                st.write(f"\nLLM Suggestion for general error in {trigger[0]}:\n{suggestion}")

    except Exception as e:
        st.write(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()  # Print the traceback for more details

    finally:
        if oracle_conn:
            oracle_conn.close()
        if pg_conn:
            pg_conn.close()
def Load_LLM():
    from langchain_community.llms import LlamaCpp
    path=r"C:\Users\Admin\Downloads\llama-2-7b-chat.Q4_K_M.gguf"
    # path=r"C:\Users\aipro\Documents\Promptora\Research\Migration\enc2\models\llama-2-7b-chat.Q4_K_M.gguf"
    n_gpu_layers = 32
    n_batch = 512
    n_threads=4
    llm = LlamaCpp(
            model_path=path,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=4096,
            temperature=0.8,
            repeat_penalty=1.18,
            top_p=1,
            top_k=3,
            max_tokens=512,
            verbose=False,
        )
    x11=llm.invoke("Write a 100 words essay on learning")
    