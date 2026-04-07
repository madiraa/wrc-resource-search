"""
Database module for WRC Resource Digitization System

This module handles the creation and management of the SQLite database
for storing digitized resource information.
"""

import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any
import json


class ResourceDatabase:
    """Manages the SQLite database for WRC resources."""
    
    def __init__(self, db_path: str = "wrc_resources.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            
    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def create_tables(self):
        """Create all necessary database tables."""
        
        # Images table - stores raw image data and OCR text
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                binder_name TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                ocr_text TEXT,
                ocr_confidence REAL,
                processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_hash TEXT,
                notes TEXT
            )
        """)
        
        # Resources table - structured resource information
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS resources (
                resource_id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                organization_name TEXT,
                resource_type TEXT,
                description TEXT,
                address TEXT,
                phone TEXT,
                email TEXT,
                website TEXT,
                hours TEXT,
                eligibility TEXT,
                services TEXT,
                is_current BOOLEAN DEFAULT 1,
                currency_score REAL,
                last_verified_date DATE,
                extracted_dates TEXT,  -- JSON array of dates found
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images(image_id)
            )
        """)
        
        # Resource categories table (many-to-many relationship)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS resource_categories (
                resource_id INTEGER,
                category TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                PRIMARY KEY (resource_id, category),
                FOREIGN KEY (resource_id) REFERENCES resources(resource_id)
            )
        """)
        
        # Embeddings table - for RAG/vector search
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                resource_id INTEGER NOT NULL,
                embedding_text TEXT NOT NULL,
                embedding_vector TEXT NOT NULL,  -- Stored as JSON
                embedding_model TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (resource_id) REFERENCES resources(resource_id)
            )
        """)
        
        # Processing log table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                process_type TEXT,
                status TEXT,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images(image_id)
            )
        """)
        
        # Create indexes for better query performance
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_resources_type 
            ON resources(resource_type)
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_resources_current 
            ON resources(is_current)
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_images_binder 
            ON images(binder_name)
        """)
        
        self.conn.commit()
        
    def insert_image(self, file_path: str, binder_name: str, 
                     original_filename: str, ocr_text: str = None,
                     ocr_confidence: float = None, image_hash: str = None) -> int:
        """Insert a new image record."""
        self.cursor.execute("""
            INSERT INTO images (file_path, binder_name, original_filename, 
                              ocr_text, ocr_confidence, image_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (file_path, binder_name, original_filename, ocr_text, 
              ocr_confidence, image_hash))
        self.conn.commit()
        return self.cursor.lastrowid
        
    def insert_resource(self, image_id: int, **kwargs) -> int:
        """Insert a new resource record."""
        fields = ['image_id']
        values = [image_id]
        
        allowed_fields = ['organization_name', 'resource_type', 'description',
                         'address', 'phone', 'email', 'website', 'hours',
                         'eligibility', 'services', 'is_current', 'currency_score',
                         'last_verified_date', 'extracted_dates']
        
        for field in allowed_fields:
            if field in kwargs:
                fields.append(field)
                values.append(kwargs[field])
        
        placeholders = ', '.join(['?'] * len(fields))
        field_names = ', '.join(fields)
        
        self.cursor.execute(f"""
            INSERT INTO resources ({field_names})
            VALUES ({placeholders})
        """, values)
        self.conn.commit()
        return self.cursor.lastrowid
        
    def add_resource_category(self, resource_id: int, category: str, 
                             confidence: float = 1.0):
        """Add a category to a resource."""
        self.cursor.execute("""
            INSERT OR REPLACE INTO resource_categories 
            (resource_id, category, confidence)
            VALUES (?, ?, ?)
        """, (resource_id, category, confidence))
        self.conn.commit()
        
    def insert_embedding(self, resource_id: int, embedding_text: str,
                        embedding_vector: List[float], model: str):
        """Insert an embedding for RAG."""
        vector_json = json.dumps(embedding_vector)
        self.cursor.execute("""
            INSERT INTO embeddings (resource_id, embedding_text, 
                                   embedding_vector, embedding_model)
            VALUES (?, ?, ?, ?)
        """, (resource_id, embedding_text, vector_json, model))
        self.conn.commit()
        return self.cursor.lastrowid
        
    def log_processing(self, image_id: Optional[int], process_type: str,
                      status: str, message: str = ""):
        """Log a processing event."""
        self.cursor.execute("""
            INSERT INTO processing_log (image_id, process_type, status, message)
            VALUES (?, ?, ?, ?)
        """, (image_id, process_type, status, message))
        self.conn.commit()
        
    def get_all_resources(self, current_only: bool = False) -> List[Dict]:
        """Retrieve all resources with their associated data."""
        query = """
            SELECT 
                r.*,
                i.file_path,
                i.binder_name,
                i.ocr_text,
                GROUP_CONCAT(rc.category, ', ') as categories
            FROM resources r
            JOIN images i ON r.image_id = i.image_id
            LEFT JOIN resource_categories rc ON r.resource_id = rc.resource_id
        """
        
        if current_only:
            query += " WHERE r.is_current = 1"
            
        query += " GROUP BY r.resource_id"
        
        self.cursor.execute(query)
        return [dict(row) for row in self.cursor.fetchall()]
        
    def search_resources(self, keyword: str = None, resource_type: str = None,
                        current_only: bool = True) -> List[Dict]:
        """Search resources by keyword and/or type."""
        query = """
            SELECT 
                r.*,
                i.file_path,
                i.binder_name,
                i.ocr_text,
                i.original_filename,
                GROUP_CONCAT(rc.category, ', ') as categories
            FROM resources r
            JOIN images i ON r.image_id = i.image_id
            LEFT JOIN resource_categories rc ON r.resource_id = rc.resource_id
            WHERE 1=1
        """
        params = []
        
        if current_only:
            query += " AND r.is_current = 1"
            
        if keyword:
            query += """ AND (
                r.organization_name LIKE ? OR
                r.description LIKE ? OR
                r.services LIKE ? OR
                i.ocr_text LIKE ?
            )"""
            keyword_param = f"%{keyword}%"
            params.extend([keyword_param] * 4)
            
        if resource_type:
            query += " AND r.resource_type = ?"
            params.append(resource_type)
            
        query += " GROUP BY r.resource_id"
        
        self.cursor.execute(query, params)
        return [dict(row) for row in self.cursor.fetchall()]
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}
        
        # Total images
        self.cursor.execute("SELECT COUNT(*) as count FROM images")
        stats['total_images'] = self.cursor.fetchone()['count']
        
        # Total resources
        self.cursor.execute("SELECT COUNT(*) as count FROM resources")
        stats['total_resources'] = self.cursor.fetchone()['count']
        
        # Current vs outdated
        self.cursor.execute("""
            SELECT is_current, COUNT(*) as count 
            FROM resources 
            GROUP BY is_current
        """)
        for row in self.cursor.fetchall():
            key = 'current_resources' if row['is_current'] else 'outdated_resources'
            stats[key] = row['count']
            
        # Resources by type
        self.cursor.execute("""
            SELECT resource_type, COUNT(*) as count 
            FROM resources 
            WHERE resource_type IS NOT NULL
            GROUP BY resource_type
        """)
        stats['by_type'] = {row['resource_type']: row['count'] 
                           for row in self.cursor.fetchall()}
        
        return stats

