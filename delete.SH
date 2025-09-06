cd /home/root/app 
docker exec -it mongodb_prod mongosh
use zhimo
db.current_hot_news.deleteMany({})
db.hot_news_metadata.deleteMany({})
db.hot_news_processed.deleteMany({})
db.news_analysis_queue.deleteMany({})
db.news_analysis_records.deleteMany({})
db.news_daily_analysis.deleteMany({})
db.news_data.deleteMany({})
db.news_hourly.deleteMany({})
db.transformed_news.deleteMany({})
db.transformed_news.find().limit(10).pretty()
db.news_analysis_queue.find().limit(10).pretty()
db.hot_news_metadata.countDocuments({})
db.hot_news_processed.countDocuments({})
db.transformed_news.countDocuments({})


db.hot_news_processed.find().sort({ heat_sum: -1 }).limit(10).pretty()
db.hot_news_processed.find().limit(10).pretty()
db.current_hot_news.find().limit(10).pretty()
db.transformed_news.find().sort({ analyzed_at: -1 }).limit(1).pretty()