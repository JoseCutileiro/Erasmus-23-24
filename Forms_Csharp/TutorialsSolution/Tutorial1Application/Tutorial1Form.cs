using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Tutorial1Application
{
    public partial class Tutorial1Form : Form
    {
        public Tutorial1Form()
        {
            InitializeComponent();
            setRandButton.Click += new EventHandler(HandleRandButtonClick);
        }

        private void Tutorial1Form_Load(object sender, EventArgs e)
        {

        }

        private void menuStrip_ItemClicked(object sender, ToolStripItemClickedEventArgs e)
        {

        }

        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        private void fileToolStripMenuItem_Click(object sender, EventArgs e)
        {

        }

        private void setBlueButton_Click(object sender, EventArgs e)
        {
            this.BackColor = Color.Blue;
        }

        private void SetGreenButton_Click(object sender, EventArgs e)
        {
            this.BackColor = Color.Green;
        }
        private void HandleRandButtonClick(object sender, EventArgs e)
        {
            Random rnd = new Random();

            int r = rnd.Next(0, 255);
            int b = rnd.Next(0, 255);
            int g = rnd.Next(0, 255);

            this.BackColor = Color.FromArgb(r,g,b);
        }

    }
}
