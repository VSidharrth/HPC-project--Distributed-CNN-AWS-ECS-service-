# Use official Nginx image
FROM nginx:alpine

# Copy HTML and model folder into Nginx's web root
COPY ./index.html /usr/share/nginx/html/
COPY ./webs_model /usr/share/nginx/html/webs_model

# Expose port 80
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]