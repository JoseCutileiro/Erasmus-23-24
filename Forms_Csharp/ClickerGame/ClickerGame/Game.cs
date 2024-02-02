using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ClickerGame
{
    public partial class Game : Form
    {
        private int points = 0;
        private int inc = 1;

        private void UpdatePointsLabel()
        {
            Points.Text = $"Points: {points}";
            Inc.Text = $"Increment: {inc}";
        }

        public Game()
        {
            InitializeComponent();
            UpdatePointsLabel();
        }

        private void F_Load(object sender, EventArgs e)
        {

        }

        private void Points_Click(object sender, EventArgs e)
        {

        }

        private void Add_Click(object sender, EventArgs e)
        {
            points += inc;
            UpdatePointsLabel();
        }

        private void IncBonus_Click(object sender, EventArgs e)
        {
            if (points > 100)
            {
                points -= 100;
                inc += 1;
            }
            UpdatePointsLabel();
        }
    }
}
