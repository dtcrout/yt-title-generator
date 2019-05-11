# META ]------------------------------------------------------------------------
help:
	@echo "test				Return status of source files."
	@echo "deps     			Install dependencies."
	@echo "thumbnails			Download YouTube thumbnails from metadata."
	@echo "thumbnails			Create YouTube thumbnails titles dictionary."
	@echo "features			Download VGG16 image features for thumbnails."
	@echo "clean				Remove artifacts and standardize repo."

# CORE ]------------------------------------------------------------------------
test:
	black --check app/src/

deps:
	pip3 install -r requirements.txt

thumbnails:
	mkdir -p app/resources/thumbnails && python3 app/src/training_data.py thumbnails

titles:
	python3 app/src/training_data.py titles

features:
	python3 app/src/training_data.py features

clean:
	black app/src/
