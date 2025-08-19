Minimal fix (make v4 behave like v3)

In get_frequent_columns_v4.py, change _extract_query_logs to select the same 3 fields as v3, and keep your table filter:

# get_frequent_columns_v4.py

def _extract_query_logs(self, table_name: str, months_back: int) -> List[Dict[str, Any]]:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)

    table_simple = table_name.split('.')[-1].lower()  # catch unqualified uses

    query = f"""
    SELECT
        dwh_business_date,
        client_id,
        SUBSTR(query_text, 1, 300) as query_text
    FROM gfolydal_managed.jdbc_centralized_audit_log
    WHERE
        dwh_business_date >= {start_date.strftime('%Y%m%d')}
        AND dwh_business_date <= {end_date.strftime('%Y%m%d')}
        AND LOWER(query_text) LIKE '%select%'
        AND (
            LOWER(query_text) LIKE '%{table_name.lower()}%'  -- fully-qualified
            OR LOWER(query_text) LIKE '%{table_simple}%'     -- simple table name
        )
        AND user_action = 'Original Query'
        AND client_id NOT LIKE '%fid%'
    LIMIT 200
    """
    ...


Why:

Restoring the three columns aligns with the v4 parser’s len(parts) >= 3 logic. 

Adding the table_simple OR-condition makes matches robust even when users don’t fully qualify the table in their SQL (v4 currently requires the fully-qualified string). 

Defensive improvement (optional but recommended)

Even with the fix above, make the parser tolerant of a 1-column export (future-proofing):

# get_frequent_columns_v4.py inside _parse_log_file loop, right after parts = self._split_csv_line(line)

if len(parts) >= 3:
    log_entry = {
        'dwh_business_date': self._clean_text(parts[0]),
        'client_id': self._clean_text(parts[1]),
        'query_text': self._clean_text(parts[2]),
    }
elif len(parts) == 1:
    # Fallback: treat the single field as the query_text
    log_entry = {
        'dwh_business_date': '',
        'client_id': '',
        'query_text': self._clean_text(parts[0]),
    }
else:
    continue


This small guard lets v4 survive if someone later changes the SQL again to only return query_text while keeping the same parser path (today it hard-requires 3 fields). The current code drops anything with < 3 parts.