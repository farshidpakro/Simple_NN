using System.Reflection;
using System.Diagnostics;

namespace UploadPic;

public partial class Form1 : Form
{
    private Button button1;



    public Form1()
    {
        InitializeComponent();
        this.Size = new System.Drawing.Size(600, 400);
        this.StartPosition = FormStartPosition.CenterScreen;
        button1 = new Button();
        this.Controls.Add(button1);
        button1.Click += new EventHandler(button1_Click);
        button1.Text = "Select Picture File";
        button1.Location = new Point(70, 70);
        button1.Size = new Size(500, 100);
        button1.Location = new Point(40, 50);

    }
    private void button1_Click(object sender, EventArgs e)
    {
        OpenFileDialog openFileDialog = new OpenFileDialog();
        openFileDialog.Filter = "Image Files|*.jpg;*.jpeg;*.png"; // Filter to allow only image files
        if (openFileDialog.ShowDialog() == DialogResult.OK)
        {
            string selectedFile = openFileDialog.FileName;
            string currentLocation = System.AppContext.BaseDirectory;
            string fileName = currentLocation + "..\\..\\..\\..\\main.py " + selectedFile;

            Process p = new Process();
            p.StartInfo = new ProcessStartInfo(@"python", fileName)
            {
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            p.Start();

            string output = p.StandardOutput.ReadToEnd();
            p.WaitForExit();

            string sample = "The predicted";
            int index = output.IndexOf(sample);

            if (index != -1)
            {
                output = output.Substring(index + sample.Length + 1);
                MessageBox.Show(output);
            }
            else
            {
                MessageBox.Show("not match to any number");
            }


        }



    }

}
