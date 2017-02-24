import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

def computeError(model: MatrixFactorizationModel, ratings: RDD[Rating]): Double = {
    val predictions = model.predict(ratings.map(x => (x.user,x.product)))
    val predAndRating = predictions.map(x => ((x.user, x.product), x.rating)).join(ratings.map(x => ((x.user, x.product), x.rating))).values
    predAndRating.map(x => (x._1-x._2)*(x._1-x._2)).reduce(_+_)
}

val data = sc.textFile("wasbs:///train_0.txt")
var ratings = data.map(_.split("\t") match { case Array(user, item, rate) => Rating(user.toInt, item.toInt, rate.toDouble)})

val rank = 20
val numIterations = 15
val lambda = 0.05
println("Using rank: "+rank+" and "+numIterations+" iterations")
val before = System.currentTimeMillis
val model = ALS.train(ratings, rank, numIterations, lambda)
val time = System.currentTimeMillis - before

println("Time taken: "+time/1000.0)

val train_error = computeError(model, ratings)
val train_rmse = math.sqrt(train_error/ratings.count)

println("Train Error: "+train_error+", Train Rmse: "+train_rmse)

val uF_sum = model.userFeatures.map{case(idx,vec)=>(idx,vec.map(t=>t*t).reduce(_+_))}.values.reduce(_+_)
val pf_sum = model.productFeatures.map{case(idx,vec)=>(idx,vec.map(t=>t*t).reduce(_+_))}.values.reduce(_+_)
println("User sum: "+uF_sum+", product sum: "+pf_sum+", sum is: "+(uF_sum+pf_sum))

println("Test ratings: "+test_ratings.count)

val test_data = sc.textFile("wasbs:///test_0.txt")
var test_ratings = test_data.map(_.split("\t") match { case Array(user, item, rate) => Rating(user.toInt, item.toInt, rate.toDouble)})
test_ratings.cache()

val test_error = computeError(model, test_ratings)
val test_rmse = math.sqrt(test_error/test_ratings.count)

println("Test Error: "+test_error+" Test Rmse: "+test_rmse) 
