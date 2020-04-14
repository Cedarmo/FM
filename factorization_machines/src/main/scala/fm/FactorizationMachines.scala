package fm

import java.util.Random
import org.apache.spark.rdd.RDD
import scala.collection.mutable
import breeze.linalg.DenseVector
import org.apache.spark.storage.StorageLevel

class FactorizationMachines extends Serializable {

  var factorNum = 4
  var alpha = 0.1
  var beta = 1.0
  var lambda1 = 0.0
  var lambda2 = 1.0
  var wTruncated = 0.0
  var isNoShowLoss = false
  var iterationNum = 5
  var numPartitions = 200

  /**
    * factorNum
    *
    * */
  def setFactorNum(value:Int):this.type = {
    require(value > 0, s"factorNum must be more than 0, but it is $value")
    factorNum = value
    this
  }

  /**
    * alpha
    *
    * */
  def setAlpha(value:Double):this.type = {
    require(value > 0.0, s"alpha must be more than 0, but it is $value")
    alpha = value
    this
  }

  /**
    * beta
    *
    * */
  def setBeta(value:Double):this.type = {
    require(value > 0.0, s"beta must be more than 0, but it is $value")
    beta = value
    this
  }

  /**
    * L1正则化
    *
    * */
  def setLambda1(value:Double):this.type = {
    require(value >= 0.0, s"lambda1 must be not less than 0, but it is $value")
    lambda1 = value
    this
  }

  /**
    * L2正则化
    *
    * */
  def setLambda2(value:Double):this.type = {
    require(value >= 0.0, s"lambda2 must be not less than 0, but it is $value")
    lambda2 = value
    this
  }

  /**
    * w权重截断(其中不包括隐语义的权重), 主要是人工截断权重绝对值较小的值
    *
    * */
  def setWTruncated(value:Double):this.type = {
    require(value >= 0.0, s"wTruncated must be not less than 0, but it is $value")
    wTruncated = value
    this
  }

  /**
    * 是否显示损失, 不建议显示损失值, 这样增加计算量
    *
    * */
  def setIsNoShowLoss(value:Boolean):this.type = {
    isNoShowLoss = value
    this
  }

  /**
    * 迭代次数
    *
    * */
  def setIterationNum(value:Int):this.type = {
    require(value > 0, s"iterationNum must be more than 0, but it is $value")
    iterationNum = value
    this
  }

  /**
    * 分区数量
    *
    * */
  def setNumPartitions(value:Int):this.type = {
    require(value > 0, s"numPartitions must be more than 0, but it is $value")
    numPartitions = value
    this
  }

  /**
    * 训练
    *
    * dataSet: RDD[(Double, String)]    第一个是label, 第二个是用","隔开的多个特征, 特征名与特征值用"@"隔开(特征是: 特征名@特征值, 可以是离散特征也可以是连续特征, 其中离散特征的特征值为1.0)
    *
    * fm算法隐语义向量的初始化不能为0, 否则无法更新
    * */
  def fit(dataSet: RDD[(Double, String)]): RDD[(String, String)] = {
    val dataSetFormat = dataSet.map{ case (label, features) =>
      val featuresFormat = features.split(",", -1).zipWithIndex
        .map{ case (feature, index) => if (feature.nonEmpty) {s"${index+1}#$feature"}else{""} }
        .mkString(",") + ",bias@1.0"
      (label, featuresFormat)
    }
    val dataSetRepartition = dataSetFormat.map(k => ((new Random).nextInt(numPartitions), k))
      .repartition(numPartitions).map(k => k._2)
    dataSetRepartition.persist(StorageLevel.MEMORY_AND_DISK)

    val sc = dataSetRepartition.sparkContext
    val factorNumBroadcast = sc.broadcast(factorNum)
    val alphaBroadcast = sc.broadcast(alpha)
    val betaBroadcast = sc.broadcast(beta)
    val lambda1Broadcast = sc.broadcast(lambda1)
    val lambda2Broadcast = sc.broadcast(lambda2)
    val isNoShowLossBroadcast = sc.broadcast(isNoShowLoss)
    val wTruncatedBroadcast = sc.broadcast(wTruncated)

    // 训练
    var wnzBroadcast = sc.broadcast(new mutable.HashMap[String, Array[Array[Double]]]())
    for (iter <- 0 until iterationNum) {
      val trainResult = dataSetRepartition.mapPartitions(dataIterator => fmTrain(dataIterator, wnzBroadcast.value, factorNumBroadcast.value,
        alphaBroadcast.value, betaBroadcast.value, lambda1Broadcast.value, lambda2Broadcast.value, isNoShowLossBroadcast.value))
      trainResult.persist(StorageLevel.MEMORY_AND_DISK)
      // 特征权重的聚类
      val trainWeightResult = trainResult.filter(k => !k._1.equals("LossValue"))
        .aggregateByKey((new DenseVector[DenseVector[Double]](Array.fill(factorNumBroadcast.value+1, 3)(0.0).map(k => new DenseVector[Double](k))), 0L))(
          (vector, array) => (vector._1 + new DenseVector[DenseVector[Double]](array._1.map(k => new DenseVector[Double](k))), vector._2 + array._2),
          (vector1, vector2) => (vector1._1 + vector2._1, vector1._2 + vector2._2) )
        .map{ case (key, (wvnzSum, num)) =>
          val wvnAverage = wvnzSum.toArray.map(vector => vector.toArray.map(value => 1.0 * value/num.toDouble))     // 如果更新太慢, 可以把1.0改为一个较大的系数
          (key, wvnAverage)
        }.filter(k => math.abs(k._2.head.head) > wTruncatedBroadcast.value)
      trainWeightResult.persist(StorageLevel.MEMORY_AND_DISK)
      // 计算平均损失
      if(isNoShowLoss){
        val trainLossResult = trainResult.filter(k => k._1.equals("LossValue"))
          .aggregateByKey((new DenseVector[DenseVector[Double]](Array.fill(1, 1)(0.0).map(k => new DenseVector[Double](k))), 0L))(
            (vector, array) => (vector._1 + new DenseVector[DenseVector[Double]](array._1.map(k => new DenseVector[Double](k))), vector._2 + array._2),
            (vector1, vector2) => (vector1._1 + vector2._1, vector1._2 + vector2._2) )
          .map{ case (key, (wvnzSum, num)) =>
            val wvnAverage = wvnzSum.toArray.map(vector => vector.toArray.map(value => 1.0 * value/num.toDouble))     // 如果更新太慢, 可以把1.0改为一个较大的系数
            (key, wvnAverage)
          }.map(k => (k._1, k._2.head.head)).first()._2
        println(s"====第${iter+1}轮====平均损失:$trainLossResult====")
      }
      val trainWeightResultMap = mutable.HashMap(trainWeightResult.collectAsMap().toList:_*)
      wnzBroadcast = sc.broadcast(trainWeightResultMap)
      trainResult.unpersist()
      trainWeightResult.unpersist()
    }
    dataSetRepartition.unpersist()

    // 特征权重最后结果
    val featuresResultMap = wnzBroadcast.value.map(k => (k._1, k._2.map(array => array.mkString("@")).mkString(",")))
    val featuresResult = sc.parallelize(featuresResultMap.toList, numPartitions)
    featuresResult
  }

  /**
    * 每个分区训练, fm梯度更新方式是ftrl
    *
    * */
  def fmTrain(dataIterator: Iterator[(Double, String)],
              wvnzRaw: mutable.HashMap[String, Array[Array[Double]]],
              factorNum: Int,
              alpha: Double,
              beta: Double,
              lambda1: Double,
              lambda2: Double,
              isNoShowLoss: Boolean): Iterator[(String, (Array[Array[Double]], Long))] = {
    // key是特征id@特征值, 二维数组中第一维度表示w/v, 第二维度表示第t次w/v, 第t-1次n, 第t-1次z的值, Long类型是次数
    val wvnzUpdate = new mutable.HashMap[String, (Array[Array[Double]], Long)]()
    var lossSum = 0.0
    var lossNum = 0L
    for (labelFeatures <- dataIterator) {
      val label = labelFeatures._1
      val featuresArray = labelFeatures._2.split(",", -1).filter(_.nonEmpty)

      // 随机过滤出现次数太少的离散特征, 需要用的时候修改
      val featuresChooseArray = featuresArray.filter(feature =>
        if (wvnzRaw.contains(feature)) true
        else {
          feature match {
            case _ if feature.startsWith("first#") =>
              if ((new Random).nextDouble <= 0.5) true
              else false
            case _ if feature.startsWith("two#") =>
              if ((new Random).nextDouble <= 0.5) true
              else false
            case _ => true
          }
        }
      )

      // 特征分割
      val featuresChooseArraySplit = featuresChooseArray
        .map(k => k.split("@", -1))
        .map(k => (k.head, k(1).toDouble))

      // 第t次预测结果
      val (prediction, vsSum) = predictAndSum(featuresChooseArraySplit, wvnzRaw, factorNum)

      // 计算第t次的损失值
      if (isNoShowLoss) {
        val loss = -1.0 * math.log(1.0/(1+math.exp(math.max(math.min(-prediction * label, 20.0), -20.0))))
        lossSum = lossSum + loss
        lossNum = lossNum + 1L
      }

      // 所有特征的共同导数
      val gradientAllCommon = label * (1.0/(1+math.exp(math.max(math.min(-prediction * label, 20.0), -20.0)))-1.0)

      // 更新特征的权重和梯度
      for ((featureName, featureValue) <- featuresChooseArraySplit) {
        if(!featureName.equals("bias")) {
          val featureSum = wvnzUpdate.getOrElseUpdate(featureName, (Array.fill(factorNum+1, 3)(0.0), 0L))._1
          var featureNum = wvnzUpdate.getOrElseUpdate(featureName, (Array.fill(factorNum+1, 3)(0.0), 0L))._2

          val gradientVAll = Array.fill(factorNum)(0.0)
          for (s <- 0 until factorNum) {
            val vAll = wvnzRaw.getOrElse(featureName, Array.fill(1)(Array.fill(3)(0.0)) ++ Array.fill(factorNum)(Array.fill(3)(0.01))).tail
            val vnzParams = vAll(s)
            val vs = vnzParams.head
            gradientVAll(s) = gradientAllCommon * (featureValue * vsSum(s) - vs * featureValue * featureValue)
          }

          updateW(featureName, featureSum, wvnzRaw, gradientAllCommon * featureValue, 0, alpha, beta, lambda1, lambda2, factorNum)
          updateV(featureName, featureSum, wvnzRaw, gradientVAll, alpha, beta, lambda1, lambda2, factorNum)

          featureNum = featureNum + 1
          wvnzUpdate.put(featureName, (featureSum, featureNum))
        } else {
          // bias只有第一维度数组有效, 其他没效
          val featureSum = wvnzUpdate.getOrElseUpdate(featureName, (Array.fill(factorNum+1, 3)(0.0), 0L))._1
          var featureNum = wvnzUpdate.getOrElseUpdate(featureName, (Array.fill(factorNum+1, 3)(0.0), 0L))._2

          updateW(featureName, featureSum, wvnzRaw, gradientAllCommon * featureValue, 0, alpha, beta, lambda1, lambda2, factorNum)

          featureNum = featureNum + 1
          wvnzUpdate.put(featureName, (featureSum, featureNum))
        }
      }
    }

    wvnzUpdate.put("LossValue", (Array(Array(lossSum)), lossNum))
    wvnzUpdate.toIterator
  }


  /**
    * 计算第t次prediction
    *
    * */
  def predictAndSum(featuresChooseArraySplit: Array[(String, Double)], wvnzRaw: mutable.HashMap[String, Array[Array[Double]]], factorNum: Int): (Double, Array[Double]) = {

    // 线性部分
    val linearSumW = featuresChooseArraySplit.par.map{ case (featureName, featureValue) =>
      val wnzParams = wvnzRaw.getOrElse(featureName, Array.fill(1)(Array.fill(3)(0.0)) ++ Array.fill(factorNum)(Array.fill(3)(0.01))).head
      val w = wnzParams.head
      w * featureValue
    }.sum

    // 交互部分
    val vsSum = Array.fill(factorNum)(0.0)
    val vsSquareSum = Array.fill(factorNum)(0.0)
    for (s <- 0 until factorNum) {
      for ((featureName, featureValue) <- featuresChooseArraySplit) {
        if(!featureName.equals("bias")){
          val vnzParamsAll = wvnzRaw.getOrElse(featureName, Array.fill(1)(Array.fill(3)(0.0)) ++ Array.fill(factorNum)(Array.fill(3)(0.01))).tail
          val vnzParams = vnzParamsAll(s)
          val vs = vnzParams.head
          vsSum(s) = vsSum(s) + vs * featureValue
          vsSquareSum(s) = vsSquareSum(s) + vs * featureValue * vs * featureValue
        }
      }
    }
    val interactionSumV = 0.5 * (vsSum.map(vs => vs*vs).sum - vsSquareSum.sum)

    // 预测结果
    val prediction = linearSumW + interactionSumV

    (prediction, vsSum)
  }

  /**
    * 更新线性部分的wnz
    *
    * */
  def updateW(featureName:String, featureSum: Array[Array[Double]], wvnzRaw: mutable.HashMap[String, Array[Array[Double]]],
              gradient: Double, position: Int, alpha: Double, beta: Double, lambda1: Double, lambda2: Double, factorNum:Int): Unit = {

    // 更新第t次的n
    val oldN = wvnzRaw.getOrElse(featureName, Array.fill(1)(Array.fill(3)(0.0)) ++ Array.fill(factorNum)(Array.fill(3)(0.01)))(position)(1)
    val newN = oldN + gradient * gradient
    featureSum(position)(1) = featureSum(position)(1) + newN

    // 更新第t次的z
    var newZ = 0.0
    if (wvnzRaw.contains(featureName)) {
      val oldZ = wvnzRaw(featureName)(position)(2)
      newZ = oldZ + gradient - wvnzRaw(featureName)(position)(0) * (math.sqrt(newN) - math.sqrt(oldN)) / alpha
      featureSum(position)(2) = featureSum(position)(2) + newZ
    } else {
      val oldZ = 0.0
      newZ = oldZ + gradient - 0.0 * (math.sqrt(newN) - math.sqrt(oldN)) / alpha
      featureSum(position)(2) = featureSum(position)(2) + newZ
    }

    // 更新第t+1次的w
    val tmp1 = -1.0*(lambda2 + (beta + math.sqrt(newN))/alpha)
    val tmp2 = if (newZ > lambda1 || newZ < -lambda1) { if (newZ > 0.0) newZ - lambda1 else if (newZ < 0.0) newZ + lambda1 else 0.0 } else 0.0
    val newW = tmp2/tmp1
    featureSum(position)(0) = featureSum(position)(0) + newW

  }

  /**
    * 更新交互部分的wnz
    *
    * */
  def updateV(featureName:String, featureSum: Array[Array[Double]], wvnzRaw: mutable.HashMap[String, Array[Array[Double]]],
              gradientVAll: Array[Double], alpha: Double, beta: Double, lambda1: Double, lambda2: Double, factorNum:Int): Unit = {
    for(s <- 0 until factorNum) {
      updateW(featureName, featureSum, wvnzRaw, gradientVAll(s), s+1, alpha, beta, lambda1, lambda2, factorNum)
    }
  }

}