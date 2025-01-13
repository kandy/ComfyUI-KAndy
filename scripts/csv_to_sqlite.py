import csv
import sqlitedict
import os

# Define the path to the CSV file and the SQLite database

csv_file_path = '../ComfyUI-WD14-Tagger/models/wd-eva02-large-tagger-v3.csv'

script_dir = os.path.dirname(os.path.abspath(__file__))
sqlite_db_path = os.path.join(script_dir, '/data/wd-eva02-large-tagger-v3.sqlite')

# Open the CSV file and read its contents
with open(csv_file_path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    # Open the SQLite database
    with sqlitedict.SqliteDict(sqlite_db_path, autocommit=False) as db:
        # Initialize an auto-incrementing key
        auto_increment_key = 1
        # Initialize a counter for committing
        commit_counter = 0
        
        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Use the auto-incrementing key as the key and the row as the value
            key = str(auto_increment_key)
            value = row
            
            # Insert the row into the SQLite database
            db[key] = value
            
            # Increment the key for the next row
            auto_increment_key += 1
            commit_counter += 1
            
            # Commit every 1000 rows
            if commit_counter >= 1000:
                db.commit()
                commit_counter = 0
        
        # Final commit for any remaining rows
        db.commit()

print(f"CSV file '{csv_file_path}' has been successfully converted to SQLite database '{sqlite_db_path}'.")