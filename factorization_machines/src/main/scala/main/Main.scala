package main

import fm.FactorizationMachines
import sparkapplication.BaseSparkLocal

object Main extends BaseSparkLocal {
  def main(args:Array[String]):Unit = {
    val spark = this.basicSpark
    import spark.implicits._

    val data = List((1.0, "gds1@1.0,group1@1.0,brand1@1.0,1@0.6,1@1.2,1@3.4"), (-1.0, "gds2@1.0,group3@1.0,brand3@1.0,0@-0.6,0@-1.2,0@-3.4"), (1.0, "gds3@1.0,group1@1.0,brand2@1.0,,1@1.2,1@1.4"),
      (-1.0, ",group2@1.0,brand1@1.0,1@0.6,1@1.2,0@-3.4"), (1.0, "gds4@1.0,group2@1.0,brand4@1.0,,,1@3.4"), (-1.0, ",group1@1.0,brand3@1.0,1@0.6,0@-1.2,"))
    val dataRDD = spark.sparkContext.parallelize(data, 2)

    val fm = new FactorizationMachines()
      .setFactorNum(2)
      .setAlpha(0.1)
      .setBeta(1.0)
      .setLambda1(0)
      .setLambda2(1.0)
      .setWTruncated(0.0)
      .setIsNoShowLoss(true)
      .setIterationNum(5)
      .setNumPartitions(3)

    val featuresWeightRDD = fm.fit(dataRDD)

    featuresWeightRDD.foreach(println)

  }
}