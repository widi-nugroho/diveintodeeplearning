package linearregression.dataset

import ai.djl.basicdataset.cv.classification.FashionMnist
import ai.djl.ndarray.NDManager
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.dataset.Dataset
import ai.djl.translate.TranslateException
import java.awt.*
import java.awt.image.BufferedImage
import java.io.IOException
import javax.swing.BoxLayout
import javax.swing.JFrame
import javax.swing.JLabel
import javax.swing.JPanel


// Saved in the FashionMnistUtils class for later use
@Throws(IOException::class, TranslateException::class)
fun getDataset(
    usage: Dataset.Usage?,
    batchSize: Int,
    randomShuffle: Boolean
): ArrayDataset {
    val fashionMnist: FashionMnist = FashionMnist.builder().optUsage(usage)
        .setSampling(batchSize, randomShuffle)
        .build()
    fashionMnist.prepare()
    return fashionMnist
}

// Saved in the FashionMnist class for later use
fun getFashionMnistLabels(labelIndices: IntArray): Array<String?> {
    val textLabels = arrayOf(
        "t-shirt", "trouser", "pullover", "dress", "coat",
        "sandal", "shirt", "sneaker", "bag", "ankle boot"
    )
    val convertedLabels = arrayOfNulls<String>(labelIndices.size)
    for (i in labelIndices.indices) {
        convertedLabels[i] = textLabels[labelIndices[i]]
    }
    return convertedLabels
}

fun getFashionMnistLabel(labelIndice: Int): String {
    val textLabels = arrayOf(
        "t-shirt", "trouser", "pullover", "dress", "coat",
        "sandal", "shirt", "sneaker", "bag", "ankle boot"
    )
    return textLabels[labelIndice]
}

class ImagePanel : JPanel {
    var SCALE: Int
    var img: BufferedImage? = null

    constructor() {
        SCALE = 1
    }

    constructor(scale: Int, img: BufferedImage?) {
        SCALE = scale
        this.img = img
    }

    override fun paintComponent(g: Graphics) {
        val g2d = g as Graphics2D
        g2d.scale(SCALE.toDouble(), SCALE.toDouble())
        g2d.drawImage(img, 0, 0, this)
    }
}

class Container : JPanel {
    constructor(label: String?) {
        layout = BoxLayout(this, BoxLayout.Y_AXIS)
        val l = JLabel(label, JLabel.CENTER)
        l.alignmentX = Component.CENTER_ALIGNMENT
        add(l)
    }

    constructor(trueLabel: String?, predLabel: String?) {
        layout = BoxLayout(this, BoxLayout.Y_AXIS)
        val l = JLabel(trueLabel, JLabel.CENTER)
        l.alignmentX = Component.CENTER_ALIGNMENT
        add(l)
        val l2 = JLabel(predLabel, JLabel.CENTER)
        l2.alignmentX = Component.CENTER_ALIGNMENT
        add(l2)
    }
}

// Saved in the FashionMnistUtils class for later use
@Throws(IOException::class, TranslateException::class)
fun showImages(
    dataset: ArrayDataset,
    number: Int, WIDTH: Int, HEIGHT: Int, SCALE: Int,
    manager: NDManager?
) {
    // Plot a list of images
    val frame = JFrame("Fashion Mnist")
    for (record in 0 until number) {
        val X = dataset[manager, record.toLong()].data[0].squeeze(-1)
        val y = dataset[manager, record.toLong()].labels[0].getFloat().toInt()
        val img = BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_BYTE_GRAY)
        val g = img.graphics as Graphics2D
        for (i in 0 until WIDTH) {
            for (j in 0 until HEIGHT) {
                val c = X.getFloat(j.toLong(), i.toLong()) / 255 // scale down to between 0 and 1
                g.color = Color(c, c, c) // set as a gray color
                g.fillRect(i, j, 1, 1)
            }
        }
        val panel: JPanel = ImagePanel(SCALE, img)
        panel.preferredSize = Dimension(WIDTH * SCALE, HEIGHT * SCALE)
        val container: JPanel = Container(getFashionMnistLabel(y))
        container.add(panel)
        frame.contentPane.add(container)
    }
    frame.contentPane.layout = FlowLayout()
    frame.pack()
    frame.isVisible = true
}

fun main(){
    val batchSize = 256
    val randomShuffle = true

    val mnistTrain = getDataset(Dataset.Usage.TRAIN, batchSize, randomShuffle)
    val mnistTest = getDataset(Dataset.Usage.TEST, batchSize, randomShuffle)

    val manager = NDManager.newBaseManager()

    System.out.println(mnistTrain.size());
    System.out.println(mnistTest.size());

    var SCALE:Int = 4;
    var WIDTH:Int = 28;
    var HEIGHT:Int = 28;

    for (batch in mnistTrain.getData(manager)) {
        val X = batch.data.head()
        val y = batch.labels.head()
        continue
    }

    /* Uncomment the following line and run to display images.
   It will open in another window. */
   showImages(mnistTrain, 18, WIDTH, HEIGHT, SCALE, manager);
}