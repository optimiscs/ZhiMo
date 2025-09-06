docker-compose -f docker-compose.prod.yml logs nginx
docker-compose -f docker-compose.prod.yml logs -f backend
docker-compose -f docker-compose.prod.yml logs -f db
docker-compose -f docker-compose.prod.yml logs -f redis
docker-compose -f docker-compose.prod.yml logs -f celeryworker
docker-compose -f docker-compose.prod.yml logs -f celerybeat