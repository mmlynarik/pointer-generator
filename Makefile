.PHONY: django, dramatiq, dramatiqr, test migrate, makemigrations, static, db, loadprod, loadtest, flush, superuser, venv, venvd, ssh, git, poetry, repo


##### APP SCRIPTS #####
train:
	train_word2vec_model -s 2011-01-01 -e 2019-12-31 -v 0 -t 0

wordsim:
	test_word_similarity


##### DEV & DEPLOY #####
test:
	python -m unittest discover -s tests -t .

precommit:
	pre-commit run --all-files


image:
	docker build --build-arg HTTP_PROXY=$$HTTP_PROXY --build-arg HTTPS_PROXY=$$HTTP_PROXY -t nlp:latest .

bash:
	docker run -it --name nlp nlp bash

##### DEV DATABASE MNGM ####
db:
	docker run -d --name postgres -e POSTGRES_USER=$${POSTGRES_USER} -e POSTGRES_PASSWORD=$${POSTGRES_PASSWORD} -p 5432:5432 -v ${HOME}/data:/var/lib/postgresql/data postgres:15
	sleep 2
	cd src/djangoproject/; \
	python manage.py migrate
	load_train_reviews_data

dbd:
	sudo rm -rf ~/data
	docker rm -f postgres


##### REPOSITORY & VENV #####
ssh:
	ssh-keygen -t rsa -b 4096 -C "miroslav.mlynarik@gmail.com" -N '' -f ~/.ssh/id_rsa
	cat ~/.ssh/id_rsa.pub

git:
	git config --global user.name "Miroslav Mlynarik"
	git config --global user.email "miroslav.mlynarik@gmail.com"
	git config --global remote.origin.prune true

poetry:
	curl -sSL https://install.python-poetry.org | python3.9 -

venv:
	poetry config virtualenvs.in-project true
	python3.9 -m venv .venv; \
	cp .env_tmpl .env; \
	echo "set -a && . ./.env && set +a" >> .venv/bin/activate; \
	. .venv/bin/activate; \
	pip install -U pip setuptools wheel; \
	sudo apt install libpq-dev; \
	poetry install

venvd:
	rm -rf .venv


##### DJANGO #####
shell:
	cd src/djangoproject; \
	python manage.py shell

app:
	cd src/djangoproject/; \
	python manage.py collectstatic; \
	python manage.py migrate; \
	python manage.py createsuperuser

django:
	cd src/djangoproject/; \
	python manage.py runserver

superuser:
	cd src/djangoproject/; \
	python manage.py createsuperuser

migrate:
	cd src/djangoproject/; \
	python manage.py migrate

makemigrations:
	cd src/djangoproject; \
	python manage.py makemigrations

static:
	cd src/djangoproject/; \
	python manage.py collectstatic

loaddata:
	cd src/djangoproject/; \
	python manage.py migrate; \
	python manage.py flush --no-input; \
	python manage.py loaddata db_backup.json

flush:
	cd djangoproject/; \
	python manage.py flush


##### CLI PRETTY #####
posh:
	mkdir ~/.poshthemes/
	wget https://github.com/JanDeDobbeleer/oh-my-posh/releases/latest/download/posh-linux-amd64
	sudo mv posh-linux-amd64 /usr/local/bin/oh-my-posh
	wget https://raw.githubusercontent.com/mmlynarik/python/master/config/paradox.omp.json
	mv paradox.omp.json ~/.poshthemes/paradox.omp.json
	sudo chmod +x /usr/local/bin/oh-my-posh
	echo eval "$$(sudo oh-my-posh --init --shell bash --config ~/.poshthemes/paradox.omp.json)" >> ~/.bashrc
