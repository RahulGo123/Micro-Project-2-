CREATE TABLE IF NOT EXISTS satellite_data (
    image_id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    ground_truth_label VARCHAR(50) NOT NULL, 
    source_satellite VARCHAR(50),          
    capture_date DATE,
    resolution_meters INT,                  
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);