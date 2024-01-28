using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Text;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using GeographyLibrary;

namespace Tutorial2Solution
{
    public partial class Form1 : Form
    {

        private List<Country> countryList;

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            
        }

        private void button1_Click_1(object sender, EventArgs e)
        {
            // INIT

            countryList = new List<Country>();

            Country c1 = new Country();
            c1.Name = "Portugal";
            c1.Population = 11;

            Country c2 = new Country();
            c2.Name = "Sweden";
            c2.Population = 10;

            countryList.Add(c1);
            countryList.Add(c2);

            button2.Enabled = true;

        }

        private void button2_Click(object sender, EventArgs e)
        {
            // SORT
            countryList.Sort((a, b) => a.Population.CompareTo(b.Population));
            listBox1.Items.Clear();

            foreach (Country c in countryList)
            {
                string info = "[" + c.Name + "]: Population " + c.Population + " millions";
                listBox1.Items.Add(info);
            }
        }

        private void listBox1_SelectedIndexChanged(object sender, EventArgs e)
        {

        }
    }
}
