import os
import gradio as gr
import sqlite3
import requests
import json
import time
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional

# Load environment variables from .env file
load_dotenv()

# Fetch the Grok API key from environment variables
GROK_API_KEY = os.getenv('GROK_API_KEY')
if GROK_API_KEY is None:
    raise ValueError("Grok API key is not set in environment variables.")

# Model accuracy tracking
class ModelAccuracyTracker:
    def __init__(self):
        self.total_queries = 0
        self.successful_queries = 0
        self.syntax_errors = 0
        self.semantic_errors = 0
        self.response_times = []
    
    def record_success(self, response_time):
        self.total_queries += 1
        self.successful_queries += 1
        self.response_times.append(response_time)
    
    def record_syntax_error(self, response_time):
        self.total_queries += 1
        self.syntax_errors += 1
        self.response_times.append(response_time)
    
    def record_semantic_error(self, response_time):
        self.total_queries += 1
        self.semantic_errors += 1
        self.response_times.append(response_time)
    
    def get_stats(self):
        if self.total_queries == 0:
            return "No queries processed yet"
        
        success_rate = (self.successful_queries / self.total_queries) * 100
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        stats = f"""
Model Accuracy Statistics:
-------------------------
Total queries: {self.total_queries}
Success rate: {success_rate:.2f}%
Syntax errors: {self.syntax_errors} ({(self.syntax_errors / self.total_queries) * 100:.2f}%)
Semantic errors: {self.semantic_errors} ({(self.semantic_errors / self.total_queries) * 100:.2f}%)
Average response time: {avg_response_time:.2f} seconds
"""
        return stats

# Initialize tracker
accuracy_tracker = ModelAccuracyTracker()

# Custom LLM class for Grok
class GrokLLM(LLM):
    api_key: str = GROK_API_KEY
    
    @property
    def _llm_type(self) -> str:
        return "grok"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that translates natural language to SQL. Use standard SQL syntax without any quoted prefixes like 'quoted.Name'. Use actual table and column names directly. Do not add backticks or code formatting to your SQL."},
                    {"role": "user", "content": prompt}
                ],
                "model": "llama3-8b-8192",
                "temperature": 0.1
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code != 200:
                error_detail = response.json().get('error', {}).get('message', 'Unknown error')
                raise ValueError(f"API request failed with status {response.status_code}: {error_detail}")
            
            result = response.json()["choices"][0]["message"]["content"]
            
            # Extra cleaning to remove problematic SQL patterns
            if "SELECT" in result and "FROM" in result:
                # Remove 'quoted.' prefixes that cause errors
                result = result.replace("quoted.", "")
            
            return result
        except Exception as e:
            raise ValueError(f"Error calling Groq API: {str(e)}")
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"api_key": self.api_key}

# Global variable to store the database path
db_path = None
db_type = "unknown"  # To track what kind of database we're working with

# Function to upload and set the SQLite database
def upload_database(db_file):
    global db_path, db_type
    db_path = db_file.name  # Get the path of the uploaded file
    
    # Try to determine database type by examining tables
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        # Check for common tables to guess database type
        if 'tracks' in tables and 'albums' in tables and 'artists' in tables:
            db_type = "music"
        elif 'orders' in tables and 'customers' in tables and 'products' in tables:
            db_type = "retail"
        elif 'employees' in tables and 'departments' in tables:
            db_type = "hr"
        else:
            db_type = "generic"
            
        conn.close()
    except:
        db_type = "unknown"
    
    return f"Database {db_file.name} uploaded successfully! Detected type: {db_type}"

# Function to clean SQL query by removing code blocks and problematic syntax
def clean_sql_query(sql_query):
    # Check if the query contains markdown code blocks
    if "```" in sql_query:
        # Extract content between code blocks if present
        code_start = sql_query.find("```") + 3
        # Skip the language identifier if present (like ```sql)
        if "sql" in sql_query[code_start:code_start+5].lower():
            code_start = sql_query.find("\n", code_start) + 1
        code_end = sql_query.rfind("```")
        if code_end > code_start:
            sql_query = sql_query[code_start:code_end].strip()
    
    # Remove any remaining code block markers
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    # Remove quoted. prefixes and other problematic patterns
    sql_query = sql_query.replace("quoted.", "")
    sql_query = sql_query.replace('"', '')  # Remove double quotes from column/table names
    
    # Fix common SQLite syntax issues
    # Fix GROUP BY with aggregate functions
    if "GROUP BY" in sql_query.upper() and "COUNT(" in sql_query.upper():
        # Replace GROUP BY 1, GROUP BY 2 with specific columns
        lines = sql_query.split('\n')
        select_line = ""
        for line in lines:
            if "SELECT" in line.upper():
                select_line = line
                break
                
        # If we found GROUP BY with numbers, try to replace them with actual columns
        if "GROUP BY" in sql_query.upper() and any(f"GROUP BY {i}" in sql_query.upper() for i in range(1, 10)):
            # Extract column names from SELECT
            if select_line:
                cols = []
                # Simple parser for columns
                parts = select_line.split("SELECT")[1].split(",")
                for i, part in enumerate(parts):
                    cols.append(part.strip().split(" AS ")[0].strip())
                
                # Replace GROUP BY 1, 2, etc. with actual column names
                for i in range(1, len(cols) + 1):
                    sql_query = sql_query.replace(f"GROUP BY {i}", f"GROUP BY {cols[i-1]}")
    
    return sql_query
    
# Function to run the LangChain SQL queries
def query_sql_db(query):
    global accuracy_tracker
    
    if db_path is None:
        return "Please upload an SQLite database file first.", "", "", accuracy_tracker.get_stats()

    start_time = time.time()
    
    try:
        # Connect to the uploaded SQLite database
        input_db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        
        # Get schema information to help the LLM understand the database structure
        table_info = input_db.get_table_info()
        
        # Create a more specific system prompt with schema information
        system_prompt = f"""You are a SQL query generator for SQLite databases.

The database has the following schema:
{table_info}

IMPORTANT RULES:
1. Use ONLY table names and column names EXACTLY as they appear above
2. Do NOT use quoted.Name notation or backticks in your SQL
3. Use table name prefixes for columns when joining tables
4. Do NOT put your SQL inside code blocks with ``` markers
5. Format your response exactly like:
   SQLQuery: [your SQL query here]
   SQLResult: [placeholder for result]
   Answer: [placeholder for answer]
6. For SQLite-specific syntax:
   - Do NOT use aggregate functions in GROUP BY clauses
   - Use simple column names or expressions in GROUP BY, not positional references
   - Quotes for string values must be single quotes (')
   - Do not use database-specific functions that might not exist in SQLite
"""
        
        # Create an instance of the Grok model with enhanced prompt
        grok_llm = GrokLLM()
        
        # Add context based on detected database type
        db_context = ""
        if db_type == "music":
            db_context = "This appears to be a music database with artists, albums, and tracks."
        elif db_type == "retail":
            db_context = "This appears to be a retail/sales database with customers, orders, and products."
        elif db_type == "hr":
            db_context = "This appears to be an HR database with employees and departments."
        
        # Override system message for this specific SQLDatabaseChain instance
        grok_llm._call = lambda prompt, stop=None: _custom_call(
            prompt, 
            stop=stop, 
            system_msg=system_prompt, 
            table_info=table_info,
            db_context=db_context
        )
        
        # Use the LLM instance in the SQLDatabaseChain
        db_chain = SQLDatabaseChain.from_llm(grok_llm, input_db, verbose=True)
        
        # Execute the query
        full_result = db_chain.run(query)
        
        # Parse the result to extract SQL query and result separately
        sql_query = ""
        sql_result = ""
        answer = ""
        
        if "SQLQuery:" in full_result:
            # Extract SQL query - make sure it's just the query
            query_start = full_result.find("SQLQuery:") + len("SQLQuery:")
            query_end = full_result.find("SQLResult:") if "SQLResult:" in full_result else len(full_result)
            sql_query = full_result[query_start:query_end].strip()
            
            # Clean up SQL query
            sql_query = clean_sql_query(sql_query)
            
            if "SQLResult:" in full_result:
                # Extract SQL result - make sure it's just the raw result
                result_start = full_result.find("SQLResult:") + len("SQLResult:")
                answer_start = full_result.find("Answer:") 
                if answer_start > result_start:
                    sql_result = full_result[result_start:answer_start].strip()
                    answer = full_result[answer_start + len("Answer:"):].strip()
                else:
                    sql_result = full_result[result_start:].strip()
                    # If no Answer section, use the original result but don't repeat SQL info
                    answer = "No explicit answer provided."
            else:
                answer = "No SQL result was provided."
        else:
            # If the expected format isn't found, return the full result
            answer = full_result
            
        # Record successful query
        end_time = time.time()
        accuracy_tracker.record_success(end_time - start_time)
            
        return answer, sql_query, sql_result, accuracy_tracker.get_stats()
        
    except Exception as e:
        error_message = str(e)
        end_time = time.time()
        
        # Handle SQL errors with Markdown code blocks in the query
        if ("syntax error" in error_message or "near" in error_message) and "```" in error_message:
            # Try to extract and execute just the SQL without code blocks
            try:
                sql_start = error_message.find("```") + 3
                sql_end = error_message.rfind("```")
                problematic_sql = error_message[sql_start:sql_end].strip()
                
                # Clean the SQL query
                clean_sql = clean_sql_query(problematic_sql)
                
                # Try to execute the cleaned SQL
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(clean_sql)
                result = cursor.fetchall()
                conn.close()
                
                # Format the result
                result_str = str(result)
                
                # Record successful recovery
                accuracy_tracker.record_success(end_time - start_time)
                
                return f"Successfully executed after cleanup.", clean_sql, result_str, accuracy_tracker.get_stats()
            except Exception as inner_e:
                # Record syntax error
                accuracy_tracker.record_syntax_error(end_time - start_time)
                
                return f"Error: Failed to execute even after SQL cleanup: {str(inner_e)}", "", "", accuracy_tracker.get_stats()
        
        # More user-friendly error message for common SQL errors
        if "no such column" in error_message:
            # Record semantic error
            accuracy_tracker.record_semantic_error(end_time - start_time)
            
            column_name = error_message.split("no such column:")[1].split("\n")[0].strip() if ":" in error_message else "unknown"
            return f"Error: The database doesn't have a column named {column_name}. Please try rephrasing your question.", "", "", accuracy_tracker.get_stats()
            
        elif "aggregate functions are not allowed in the GROUP BY" in error_message:
            # Record syntax error
            accuracy_tracker.record_syntax_error(end_time - start_time)
            
            # Try to extract and fix the query
            try:
                sql_start = error_message.find("[SQL:") + 6
                sql_end = error_message.rfind("]")
                problematic_sql = error_message[sql_start:sql_end].strip()
                
                # Fix the GROUP BY clause
                fixed_sql = problematic_sql
                
                # Replace GROUP BY with positional arguments
                if "GROUP BY" in fixed_sql:
                    lines = fixed_sql.split('\n')
                    select_line = ""
                    group_by_line = ""
                    
                    for line in lines:
                        if "SELECT" in line.upper():
                            select_line = line
                        if "GROUP BY" in line.upper():
                            group_by_line = line
                    
                    if select_line and group_by_line:
                        # Extract all parts after GROUP BY
                        group_by_parts = group_by_line.split("GROUP BY")[1].strip().split(",")
                        
                        # For each numeric reference, replace with actual column
                        for part in group_by_parts:
                            part = part.strip()
                            if part.isdigit():
                                # This is a positional reference, replace it
                                pos = int(part)
                                select_parts = select_line.split("SELECT")[1].split(",")
                                if 0 < pos <= len(select_parts):
                                    # Get the expression at the position
                                    expr = select_parts[pos-1].strip()
                                    # If it has an alias, use the column name before AS
                                    if " AS " in expr:
                                        expr = expr.split(" AS ")[0].strip()
                                    # Replace the numeric reference
                                    fixed_sql = fixed_sql.replace(f"GROUP BY {part}", f"GROUP BY {expr}")
                
                # Remove aggregate functions from GROUP BY
                if "GROUP BY" in fixed_sql and ("COUNT(" in fixed_sql.upper() or "SUM(" in fixed_sql.upper() or "AVG(" in fixed_sql.upper()):
                    from_pos = fixed_sql.upper().find("FROM")
                    if from_pos > 0:
                        select_clause = fixed_sql[:from_pos].strip()
                        from_clause = fixed_sql[from_pos:].strip()
                        
                        # Extract column aliases
                        columns = []
                        if "SELECT" in select_clause:
                            col_part = select_clause.split("SELECT")[1].split(",")
                            for col in col_part:
                                col = col.strip()
                                if " AS " in col:
                                    # Use the alias name
                                    alias = col.split(" AS ")[1].strip()
                                    columns.append(alias)
                                else:
                                    # Use the column name
                                    columns.append(col)
                        
                        # Find non-aggregate columns for GROUP BY
                        non_aggregate_cols = []
                        for col in columns:
                            if not any(agg in col.upper() for agg in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]):
                                non_aggregate_cols.append(col)
                        
                        # Replace GROUP BY clause with non-aggregate columns
                        if non_aggregate_cols:
                            group_by_pos = from_clause.upper().find("GROUP BY")
                            if group_by_pos > 0:
                                order_by_pos = from_clause.upper().find("ORDER BY")
                                limit_pos = from_clause.upper().find("LIMIT")
                                
                                # Find the end of GROUP BY clause
                                end_pos = min(p for p in [order_by_pos, limit_pos] if p > 0) if order_by_pos > 0 or limit_pos > 0 else len(from_clause)
                                
                                # Replace GROUP BY clause
                                new_group_by = "GROUP BY " + ", ".join(non_aggregate_cols)
                                fixed_sql = select_clause + " " + from_clause[:group_by_pos] + new_group_by + from_clause[end_pos:]
                
                # Try to execute the fixed query
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(fixed_sql)
                result = cursor.fetchall()
                conn.close()
                
                # Format the result
                result_str = str(result)
                
                # Record successful recovery
                accuracy_tracker.record_success(end_time - start_time)
                
                return f"Fixed query after removing aggregate functions from GROUP BY.", fixed_sql, result_str, accuracy_tracker.get_stats()
            except Exception as inner_e:
                # Still record syntax error
                accuracy_tracker.record_syntax_error(end_time - start_time)
                
                return f"Error: Could not fix GROUP BY with aggregates: {str(inner_e)}", "", "", accuracy_tracker.get_stats()
        else:
            # Generic error, record as syntax error
            accuracy_tracker.record_syntax_error(end_time - start_time)
            
            return f"An error occurred: {error_message}", "", "", accuracy_tracker.get_stats()

# Helper function to customize the LLM call with additional context
def _custom_call(prompt, stop=None, system_msg=None, table_info=None, db_context=""):
    try:
        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Create a more helpful user prompt by adding schema information
        enhanced_prompt = prompt
        if table_info and "schema" not in prompt.lower():
            enhanced_prompt = f"Database schema information: {table_info}\n\n{db_context}\n\n{prompt}"
        
        data = {
            "messages": [
                {"role": "system", "content": system_msg or "You are a helpful assistant that translates natural language to SQL. Do NOT use quoted.Name or similar notation in your queries. Do NOT include backticks (```) in your response."},
                {"role": "user", "content": enhanced_prompt}
            ],
            "model": "llama3-8b-8192",
            "temperature": 0.1  # Lower temperature for more deterministic output
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            error_detail = response.json().get('error', {}).get('message', 'Unknown error')
            raise ValueError(f"API request failed with status {response.status_code}: {error_detail}")
        
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise ValueError(f"Error calling Groq API: {str(e)}")

# Create a Gradio interface for both file upload and query
with gr.Blocks() as iface:
    gr.Markdown("# Text-to-SQL using Groq API")
    gr.Markdown("Upload your SQLite database and then type a question in plain English to get the SQL query result.")

    # File upload component
    db_file = gr.File(label="Upload SQLite Database", file_types=[".sqlite", ".db"])
    upload_btn = gr.Button("Upload Database")
    upload_output = gr.Textbox(label="Upload Status")

    # Input for querying the database
    query_input = gr.Textbox(label="Enter your query in plain English", placeholder="e.g., How many tracks are in the database?")
    
    # Separate outputs for SQL query and results
    final_answer_output = gr.Textbox(label="Final Answer")
    sql_query_output = gr.Textbox(label="Generated SQL Query")
    sql_result_output = gr.Textbox(label="SQL Query Results")
    accuracy_output = gr.Textbox(label="Model Accuracy Statistics")
    
    query_btn = gr.Button("Run Query")

    # Link the upload button to the upload_database function
    upload_btn.click(upload_database, inputs=db_file, outputs=upload_output)
    
    # Link the query button to the query_sql_db function with multiple outputs
    query_btn.click(query_sql_db, inputs=query_input, outputs=[final_answer_output, sql_query_output, sql_result_output, accuracy_output])

# Launch the app
iface.launch()