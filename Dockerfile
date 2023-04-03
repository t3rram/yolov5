#Base Image to use
FROM gcr.io/google-appengine/python

#Expose port 8080
EXPOSE 8501
RUN virtualenv /env -p python3

# Setting these environment variables are the same as running
# source /env/bin/activate.
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH
RUN apt-get update && apt-get install -y python-opencv
#Optional - install git to fetch packages directly from github
#RUN apt-get update && apt-get install -y git

# Copy the application's requirements.txt and run pip to install all
# dependencies into the virtualenv.
ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

#Copy Requirements.txt file into app directory
#COPY requirements.txt app/requirements.txt

#install all requirements in requirements.txt
#RUN pip install -r app/requirements.txt
ADD . /app
#Copy all files in current directory into app directory
#COPY . /app

#Change Working Directory to app directory
#WORKDIR /app

#Run the application on port 8080
#ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
CMD streamlit run --server.port 8501 --server.enableCORS false app.py