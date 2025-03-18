# Climate Research Platform

A comprehensive platform for climate data analysis, visualization, and research collaboration.

## Project Overview

This platform integrates climate data processing capabilities with interactive visualizations, allowing researchers to analyze climate patterns, trends, and make data-driven predictions. The application is containerized using Docker for easy deployment and consists of a FastAPI backend, Streamlit frontend, and PostgreSQL database.

## Project Structure

```
climate-research-platform/
├── backend/                  # Python FastAPI backend
│   ├── api/                  # API endpoints and routes
│   ├── core/                 # Core business logic
│   ├── db/                   # Database models and connections
│   ├── services/             # Service layer for business operations
│   ├── utils/                # Utility functions and helpers
│   └── requirements.txt      # Backend dependencies
├── frontend/                 # Streamlit frontend application
│   ├── pages/                # Streamlit pages
│   ├── components/           # Reusable UI components
│   ├── utils/                # Frontend utilities
│   └── requirements.txt      # Frontend dependencies
├── data/                     # Data storage
│   ├── raw/                  # Raw data files
│   │   └── combined_df_processed.csv  # Sample dataset
│   └── processed/            # Processed/transformed data
├── notebooks/                # Jupyter notebooks for analysis
│   └── analysis.ipynb        # Sample analysis notebook
├── scripts/                  # Utility scripts
├── tests/                    # Test files
│   ├── backend/              # Backend tests
│   └── frontend/             # Frontend tests
├── Dockerfile                # Backend Docker configuration
├── Dockerfile.frontend       # Frontend Docker configuration
└── docker-compose.yml        # Docker Compose configuration
```

## Prerequisites

- [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/install/)
- Python 3.8+ (for local development)

## Setup Instructions

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/climate-research-platform.git
   cd climate-research-platform
   ```

2. Build and start the Docker containers:
   ```bash
   docker-compose up --build
   ```

3. Access the applications:
   - Frontend (Streamlit): http://localhost:8501
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Local Development Setup

1. Set up the backend:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Set up the frontend:
   ```bash
   cd frontend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the backend server:
   ```bash
   cd backend
   uvicorn api.main:app --reload
   ```

4. Run the frontend application:
   ```bash
   cd frontend
   streamlit run pages/home.py
   ```

## Development

- Backend API is built with FastAPI
- Frontend is built with Streamlit
- Data processing utilizes pandas, numpy, and other scientific Python libraries
- Database models are managed with SQLAlchemy

## Contributing

1. Create a new branch for your feature
2. Implement and test your changes
3. Submit a pull request with a clear description of your changes

## License

[MIT License](LICENSE)

