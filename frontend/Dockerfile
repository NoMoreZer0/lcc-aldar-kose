# Stage 1: install dependencies and build the app (optional for dev server)
FROM node:18-bullseye AS base

WORKDIR /usr/src/app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
