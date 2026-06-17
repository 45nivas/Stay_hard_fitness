# Use a lightweight Node.js runtime as parent image
FROM node:20-alpine

# Set working directory inside the container
WORKDIR /app

# Copy package configurations
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Expose Vite development server port
EXPOSE 5173

# Start Vite dev server exposing it outside the container (host 0.0.0.0)
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
