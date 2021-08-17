package linearregression.dataset

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import org.math.plot.Plot2DPanel
import javax.swing.JFrame

class DataPoints(val x: NDArray, val y: NDArray)

// Generate y = X w + b + noise
fun syntheticData(manager: NDManager, w: NDArray, b: Float, numExamples: Long): DataPoints {
    val X = manager.randomNormal(Shape(numExamples, w.size()))
    var y = X.dot(w).add(b)
    // Add noise
    y = y.add(manager.randomNormal(0f, 0.1f, y.shape, DataType.FLOAT32, Device.defaultDevice()))
    return DataPoints(X, y)
}

fun main(){
    val manager = NDManager.newBaseManager()

    val trueW = manager.create(1f)
    val trueB = 4.2f

    val dp = syntheticData(manager, trueW, trueB, 1000)
    val features = dp.x
    val labels = dp.y
    println(features.size())
    println(labels.shape)

    //val X = features[NDIndex(":, 1")].toFloatArray()
    //val y = labels.toFloatArray()

    val x = mutableListOf<Double>()
    val y = mutableListOf<Double>()
    for (i in 0..dp.x.size()-1){
        val tx=dp.x.getFloat(i)
        x.add(tx.toDouble())
        val ty=dp.y.getFloat(i)
        y.add(ty.toDouble())
    }
    val plot = Plot2DPanel()
    plot.addScatterPlot("Plot",x.toDoubleArray(),y.toDoubleArray())
    val frame = JFrame("Linear Regression")
    frame.setSize(600, 600)
    frame.contentPane = plot
    frame.isVisible = true
}