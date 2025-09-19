FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install ALL dependencies (including devDependencies for the build)
RUN npm install

# Copy source code
# The .dockerignore file prevents the local node_modules from being copied
COPY . .

# Build the application
RUN npm run build

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Start the application using Vite's preview server
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "3000"]
