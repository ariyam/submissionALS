import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

val rank = 10
val users = sc.textFile("wasbs:///az-u60.txt").map(_.split("  ")).map(t=>(t(0).split(":")(0).toInt,t.slice(1,rank+1).map(_.toDouble)))
val items = sc.textFile("wasbs:///az-v60.txt").map(_.split("  ")).map(t=>(t(0).split(":")(0).toInt,t.slice(1,rank+1).map(_.toDouble)))
val model = new MatrixFactorizationModel(rank,users,items)

val testData = sc.textFile("wasbs:///testAmazon.dat")
var testRatings = testData.map(_.split("\t") match { case Array(user, item, rate) => Rating(user.toInt, item.toInt, rate.toDouble)})
val predictions = model.predict(testRatings.map(x => (x.user,x.product)))
val predAndRating = predictions.map(x => ((x.user, x.product), x.rating)).join(testRatings.map(x => ((x.user, x.product), x.rating))).values
val testError = predAndRating.map(x => (x._1-x._2)*(x._1-x._2)).reduce(_+_)
math.sqrt(testError/testRatings.count)
