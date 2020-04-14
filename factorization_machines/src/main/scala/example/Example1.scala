package example

import breeze.linalg.DenseVector
import sparkapplication.BaseSparkLocal

import scala.collection.mutable

object Example1 extends BaseSparkLocal {
  def main(args:Array[String]):Unit = {
//    val spark = this.basicSpark
//    import spark.implicits._

    val a = Array.fill(1)(Array.fill(3)(0.0)) ++ Array.fill(3)(Array.fill(3)(0.01))
    val d = a.map(k => k.mkString("@")).mkString(",")
    print(d)

















  }
}
