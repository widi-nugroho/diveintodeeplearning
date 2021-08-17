package mlp.activationFunction

import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.nn.Activation
import org.math.plot.Plot2DPanel
import java.awt.Color
import javax.swing.JFrame

fun main(){
    val manager = NDManager.newBaseManager()
    val x = manager.arange(-8.0f, 8.0f, 0.1f)
    x.setRequiresGradient(true)

    val gc = Engine.getInstance().newGradientCollector()
    val y: NDArray = Activation.tanh(x)
    gc.backward(y)
    gc.close()
    val res = x.gradient

    val xx = mutableListOf<Double>()
    val yy= mutableListOf<Double>()
    var dY= mutableListOf<Double>()

    for (i in 0..x.size()-1){
        xx.add(x.getFloat(i).toDouble())
        yy.add(y.getFloat(i).toDouble())
        dY.add(res.getFloat(i).toDouble())
    }

    var X: DoubleArray = xx.toDoubleArray()
    var Y: DoubleArray = yy.toDoubleArray()

    val plot = Plot2DPanel()
    plot.addLinePlot("Tanh", Color.BLACK,X,Y)
    plot.addLinePlot("dtanh", Color.BLUE,X,dY.toDoubleArray())
    plot.addLegend("NORTH")
    plot.setAxisLabel(0,"x")
    val frame = JFrame("Tanh")
    frame.setSize(600, 600)
    frame.contentPane = plot
    frame.isVisible = true
}