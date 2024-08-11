const express = require("express")
const bodyParser = require("body-parser")
const { exec } = require("child_process")
const cors = require("cors")
const { PythonShell } = require("python-shell")

const app = express()
const port = 3000

app.use(bodyParser.json())
app.use(cors())

exec(
  "pip install numpy==1.21.4 pandas==1.3.3 joblib scikit-learn==1.0.2",
  (error, stdout) => {
    console.log("Installing libraries")
    if (error) {
      console.error(`Error installing Python dependencies: ${error}`)
      return
    }
    console.log(`Python dependencies installed: ${stdout}`)
  }
)

app.post("/predict", async (req, res) => {
  const input = req.body
  console.log("Request received")

  let options = {
    mode: "text",
    pythonOptions: ["-u"], // get print results in real-time
    scriptPath: ".", // update this path
    args: [JSON.stringify(input)],
  }

  try {
    const result = await PythonShell.run("predict.py", options)
    console.log(result)
    res.json({
      prediction: result[2],
    })
  } catch (error) {
    console.log(error)
    res.status(500).json({
      prediction: "error predicting results",
    })
  }

  // .then((messages) => {
  //   console.log("message", messages)
  //   res.status(200).json({
  //     Probability: messages[2],
  //   })
  //   return
  // })
  // .catch((e) => {
  //   console.log("error", e)
  // })

  // PythonShell.run("predict.py", options, (err, result) => {
  //   if (err) {
  //     console.error("Error running Python script:", err)
  //     res.status(500).send("Error running Python script")
  //     return
  //   }
  // })
})

app.listen(port, () => {
  console.log(`Server is running on port ${port}`)
})
