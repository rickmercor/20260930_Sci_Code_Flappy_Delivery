FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gfortran && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    numpy==2.4.6 scipy==1.17.1 sympy==1.14.0 h5py==3.16.0 pandas==3.0.5 \
    scikit-learn==1.9.0 networkx==3.6.1 cvxpy==1.9.2 sfepy==2026.2
WORKDIR /app
