# ---- Build stage ----
FROM node:20-alpine AS build
WORKDIR /app

# cài deps nhanh và cache tốt
COPY package*.json ./
RUN npm install --no-audit --no-fund

# 1) cho phép nhận build-arg
ARG VITE_API_URL=/api
# 2) export vào env của process build để Vite pick up
ENV VITE_API_URL=${VITE_API_URL}

# copy source và build
COPY . .
RUN npm run build

# ---- Serve stage ----
FROM nginx:alpine
# xóa default conf, thêm conf tối ưu cho SPA
RUN rm -f /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/app.conf

# copy artifact từ stage build
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
