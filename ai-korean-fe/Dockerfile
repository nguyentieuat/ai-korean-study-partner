# ---- build stage ----
FROM node:20-alpine AS build
WORKDIR /app

# 1) cho phép nhận build-arg
ARG VITE_API_URL=/api
# 2) export vào env của process build để Vite pick up
ENV VITE_API_URL=${VITE_API_URL}

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# ---- serve stage ----
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
