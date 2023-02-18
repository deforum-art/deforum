import sys
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QKeySequence, QAction, QPainter
from PySide6.QtWidgets import QApplication, QMainWindow, QMdiArea, QMdiSubWindow, QTextEdit, QInputDialog, \
    QListWidgetItem, QPlainTextEdit, QGroupBox
import xml.etree.ElementTree as ET
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QLineEdit, QLabel, QComboBox
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QKeySequence, QMouseEvent, QDrag
from PySide6.QtWidgets import QTextEdit, QListWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsProxyWidget, QApplication
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.mdiArea = QMdiArea()
        self.setCentralWidget(self.mdiArea)

        self.newNodeEditorAct = QAction("New Node Editor", self)
        self.newNodeEditorAct.setShortcut(QKeySequence.New)
        self.newNodeEditorAct.triggered.connect(self.newNodeEditor)
        self.newNodeMakerAct = QAction("New Node Maker", self)
        self.newNodeMakerAct.triggered.connect(self.newNodeMaker)
        self.fileMenu = self.menuBar().addMenu("File")
        self.fileMenu.addAction(self.newNodeEditorAct)
        self.fileMenu.addAction(self.newNodeMakerAct)
        self.setWindowTitle("MDI Example")
        self.show()

    def newNodeEditor(self):
        editor = NodeEditor()
        sub = QMdiSubWindow()
        sub.setWidget(editor)
        sub.setAttribute(Qt.WA_DeleteOnClose)
        self.mdiArea.addSubWindow(sub)
        sub.show()

    def newNodeMaker(self):
        maker = NodeMaker()
        sub = QMdiSubWindow()
        sub.setWidget(maker)
        sub.setAttribute(Qt.WA_DeleteOnClose)
        self.mdiArea.addSubWindow(sub)
        sub.show()


class NodeMaker(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Node Maker")
        self.setMinimumSize(400, 300)

        self.nodeList = QListWidget()
        self.loadNodes()
        self.nodeList.currentItemChanged.connect(self.nodeSelected)

        self.nameEdit = QLineEdit()
        self.typeCombo = QComboBox()
        self.typeCombo.addItems(["String", "Integer", "Float", "Boolean"])
        self.addInputBtn = QPushButton("Add Input")
        self.addInputBtn.clicked.connect(self.addInput)
        self.inputList = QListWidget()
        self.removeInputBtn = QPushButton("Remove Input")
        self.removeInputBtn.clicked.connect(self.removeInput)
        self.addOutputBtn = QPushButton("Add Output")
        self.addOutputBtn.clicked.connect(self.addOutput)
        self.outputList = QListWidget()
        self.removeOutputBtn = QPushButton("Remove Output")
        self.removeOutputBtn.clicked.connect(self.removeOutput)
        self.codeEdit = QTextEdit()
        self.saveBtn = QPushButton("Save Node")
        self.saveBtn.clicked.connect(self.saveNode)

        inputOutputLayout = QVBoxLayout()
        inputOutputLayout.addWidget(QLabel("Inputs:"))
        inputOutputLayout.addWidget(self.inputList)
        inputOutputLayout.addWidget(QLabel("Outputs:"))
        inputOutputLayout.addWidget(self.outputList)

        inputOutputBtnLayout = QHBoxLayout()
        inputOutputBtnLayout.addWidget(self.nameEdit)
        inputOutputBtnLayout.addWidget(self.typeCombo)
        inputOutputBtnLayout.addWidget(self.addInputBtn)
        inputOutputBtnLayout.addWidget(self.addOutputBtn)
        inputOutputBtnLayout.addWidget(self.removeInputBtn)
        inputOutputBtnLayout.addWidget(self.removeOutputBtn)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.nodeList)
        mainLayout.addLayout(inputOutputBtnLayout)
        mainLayout.addLayout(inputOutputLayout)
        mainLayout.addWidget(QLabel("Code:"))
        mainLayout.addWidget(self.codeEdit)
        mainLayout.addWidget(self.saveBtn)

        self.setLayout(mainLayout)
        self.show()

    def loadNodes(self):
        try:
            tree = ET.parse("nodes.xml")
            root = tree.getroot()
            for child in root:
                self.nodeList.addItem(child.attrib["name"])
        except FileNotFoundError:
            pass

    @Slot()
    def nodeSelected(self):
        self.inputList.clear()
        self.outputList.clear()
        self.codeEdit.clear()
        self.nameEdit.clear()
        name = self.nodeList.currentItem().text()
        tree = ET.parse("nodes.xml")
        root = tree.getroot()
        for child in root:
            if child.attrib["name"] == name:
                self.nameEdit.setText(child.attrib["name"])
                for input in child.iter("input"):
                    self.inputList.addItem(input.attrib["name"] + " : " + input.attrib["type"])
                for output in child.iter("output"):
                    self.outputList.addItem(output.attrib["name"] + " : " + output.attrib["type"])
                self.codeEdit.setPlainText(child.find("code").text)

    @Slot()
    def addInput(self):
        name = self.nameEdit.text()
        type = self.typeCombo.currentText()
        self.inputList.addItem(name + " : " + type)

    @Slot()
    def removeInput(self):
        self.inputList.takeItem(self.inputList.currentRow())

    @Slot()
    def addOutput(self):
        name = self.nameEdit.text()
        type = self.typeCombo.currentText()
        self.outputList.addItem(name + " : " + type)

    @Slot()
    def removeOutput(self):
        self.outputList.takeItem(self.outputList.currentRow())

    @Slot()
    def saveNode(self):
        name = self.nameEdit.text()
        try:
            tree = ET.parse("nodes.xml")
            root = tree.getroot()
        except FileNotFoundError:
            root = ET.Element("nodes")
        for child in root:
            if child.attrib["name"] == name:
                root.remove(child)
        node = ET.SubElement(root, "node")
        node.set("name", name)
        for i in range(self.inputList.count()):
            input = ET.SubElement(node, "input")
            input.set("name", self.inputList.item(i).text().split(" : ")[0])
            input.set("type", self.inputList.item(i).text().split(" : ")[1])
        for i in range(self.outputList.count()):
            output = ET.SubElement(node, "output")
            output.set("name", self.outputList.item(i).text().split(" : ")[0])
            output.set("type", self.outputList.item(i).text().split(" : ")[1])
        code = ET.SubElement(node, "code")
        code.text = self.codeEdit.toPlainText()
        tree = ET.ElementTree(root)
        tree.write("nodes.xml")
        self.nodeList.clear()
        self.loadNodes()


class NodeEditor(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.nodes = {}
        self.connections = []
        self.nodeList = QListWidget()
        self.loadNodes()
        self.nodeList.itemDoubleClicked.connect(self.addNode)

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.TextAntialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        #self.setRenderHint(QPainter.HighQualityAntialiasing)
        #self.setRenderHint(QPainter.NonCosmeticDefaultPen)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.nodeList)
        mainLayout.addWidget(self.viewport())
        self.setLayout(mainLayout)
        self.show()

    def loadNodes(self):
        try:
            tree = ET.parse("nodes.xml")
            root = tree.getroot()
            for child in root:
                self.nodeList.addItem(child.attrib["name"])
        except FileNotFoundError:
            pass

    def addNode(self):
        item = self.nodeList.currentItem()
        name = item.text()
        tree = ET.parse("nodes.xml")
        root = tree.getroot()
        for child in root:
            if child.attrib["name"] == name:
                node = Node(child)
                self.nodes[name] = node
                proxy = self.scene().addWidget(node)
                proxy.setPos(QPoint(0, 0))
                node.inputsList.itemDoubleClicked.connect(self.connectInput)
                node.outputsList.itemDoubleClicked.connect(self.connectOutput)
                node.parametersList.itemChanged.connect(self.setParameter)
                node.runBtn.clicked.connect(self.runNode)
                break

    def connectInput(self):
        inputItem = self.sender().currentItem()
        inputName = inputItem.text().split(" : ")[0]
        inputType = inputItem.text().split(" : ")[1]
        outputType, outputNode, outputName = self.getOutput()
        if inputType == outputType:
            self.connections.append({"input": (inputName, self.sender().parent()), "output": (outputName, outputNode)})
            self.drawConnections()

    def getOutput(self):
        output, ok = QInputDialog.getItem(self, "Select Output", "Outputs:", [output.text() for output in self.outputsList], 0, False)
        if ok and output:
            outputName = output.split(" : ")[0]
            outputType = output.split(" : ")[1]
            return outputType, self.sender().parent(), outputName
        return None, None, None

    def connectOutput(self):
        outputItem = self.sender().currentItem()
        outputName = outputItem.text().split(" : ")[0]
        outputType = outputItem.text().split(" : ")[1]
        inputType, inputNode, inputName = self.getInput()
        if inputType == outputType:
            self.connections.append({"input": (inputName, inputNode), "output": (outputName, self.sender().parent())})
            self.drawConnections()

    def getInput(self):
        pass

    def setParameter(self):
        pass

    def runNode(self):
        pass

class Node(QGroupBox):
    def __init__(self, xmlNode):
        super().__init__()
        self.inputs = {}
        self.outputs = {}
        self.parameters = {}
        self.setTitle(xmlNode.attrib["name"])
        layout = QVBoxLayout()

        inputsLayout = QHBoxLayout()
        inputsLayout.addWidget(QLabel("Inputs:"))
        self.inputsList = QListWidget()
        inputs = xmlNode.find("inputs")
        if inputs is not None:
            for input in inputs:
                self.inputs[input.attrib["name"]] = input.attrib["type"]
                self.inputsList.addItem(input.attrib["name"] + " : " + input.attrib["type"])
        inputsLayout.addWidget(self.inputsList)
        layout.addLayout(inputsLayout)

        outputsLayout = QHBoxLayout()
        outputsLayout.addWidget(QLabel("Outputs:"))
        self.outputsList = QListWidget()
        outputs = xmlNode.find("outputs")
        if outputs is not None:
            for output in outputs:
                self.outputs[output.attrib["name"]] = output.attrib["type"]
                self.outputsList.addItem(output.attrib["name"] + " : " + output.attrib["type"])
        outputsLayout.addWidget(self.outputsList)
        layout.addLayout(outputsLayout)

        parametersLayout = QHBoxLayout()
        parametersLayout.addWidget(QLabel("Parameters:"))
        self.parametersList = QListWidget()
        parameters = xmlNode.find("parameters")
        if parameters is not None:
            for parameter in parameters:
                self.parameters[parameter.attrib["name"]] = parameter.attrib["default"]
                parameterItem = QListWidgetItem(parameter.attrib["name"] + " : " + parameter.attrib["default"])
                parameterItem.setFlags(parameterItem.flags() | Qt.ItemIsEditable)
                self.parametersList.addItem(parameterItem)
        parametersLayout.addWidget(self.parametersList)
        layout.addLayout(parametersLayout)

        codeLayout = QHBoxLayout()
        codeLayout.addWidget(QLabel("Code:"))
        self.codeEdit = QPlainTextEdit()
        code = xmlNode.find("code")
        if code is not None:
            self.codeEdit.setPlainText(code.text)
        codeLayout.addWidget(self.codeEdit)
        layout.addLayout(codeLayout)

        self.runBtn = QPushButton("Run")
        layout.addWidget(self.runBtn)

        self.setLayout(layout)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    sys.exit(app.exec_())
