import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

val data = sc.textFile("wasbs:///trainAmazon.dat")
var ratings = data.map(_.split("\t") match { case Array(user, item, rate) => Rating(user.toInt, item.toInt, rate.toDouble)})
val user_ratings = ratings.map(t=>(t.user,(t.rating,1)))

val sums_and_counts = user_ratings.reduceByKey((a,b)=>(a._1+b._1,a._2+b._2))
val avgs = sums_and_counts.mapValues(t=>t._1/t._2)

val testData = sc.textFile("wasbs:///testAmazon.dat")
var testRatings = testData.map(_.split("\t") match { case Array(user, item, rate) => (user.toInt, rate.toDouble)})

val predAndRating = testRatings.join(avgs)
predAndRating.cache()
println("joined count: "+predAndRating.count+", ratings count:"+testRatings.count)

val testError = predAndRating.values.map(x => (x._1-x._2)*(x._1-x._2)).reduce(_+_)
val testRmse = math.sqrt(testError/testRatings.count)
println("Test Error: "+testError+" Test Rmse: "+testRmse) 
