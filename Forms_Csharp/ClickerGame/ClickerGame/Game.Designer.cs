namespace ClickerGame
{
    partial class Game
    {
        /// <summary>
        /// Variável de designer necessária.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Limpar os recursos que estão sendo usados.
        /// </summary>
        /// <param name="disposing">true se for necessário descartar os recursos gerenciados; caso contrário, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Código gerado pelo Windows Form Designer

        /// <summary>
        /// Método necessário para suporte ao Designer - não modifique 
        /// o conteúdo deste método com o editor de código.
        /// </summary>
        private void InitializeComponent()
        {
            this.Points = new System.Windows.Forms.Label();
            this.Add = new System.Windows.Forms.Button();
            this.Inc = new System.Windows.Forms.Label();
            this.IncBonus = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // Points
            // 
            this.Points.AutoSize = true;
            this.Points.Location = new System.Drawing.Point(354, 172);
            this.Points.Name = "Points";
            this.Points.Size = new System.Drawing.Size(77, 16);
            this.Points.TabIndex = 0;
            this.Points.Text = "points_here";
            this.Points.Click += new System.EventHandler(this.Points_Click);
            // 
            // Add
            // 
            this.Add.Location = new System.Drawing.Point(357, 76);
            this.Add.Name = "Add";
            this.Add.Size = new System.Drawing.Size(75, 23);
            this.Add.TabIndex = 1;
            this.Add.Text = "+";
            this.Add.UseVisualStyleBackColor = true;
            this.Add.Click += new System.EventHandler(this.Add_Click);
            // 
            // Inc
            // 
            this.Inc.AutoSize = true;
            this.Inc.Location = new System.Drawing.Point(354, 200);
            this.Inc.Name = "Inc";
            this.Inc.Size = new System.Drawing.Size(58, 16);
            this.Inc.TabIndex = 2;
            this.Inc.Text = "inc_here";
            // 
            // IncBonus
            // 
            this.IncBonus.Location = new System.Drawing.Point(357, 118);
            this.IncBonus.Name = "IncBonus";
            this.IncBonus.Size = new System.Drawing.Size(75, 23);
            this.IncBonus.TabIndex = 3;
            this.IncBonus.Text = "Bonus";
            this.IncBonus.UseVisualStyleBackColor = true;
            this.IncBonus.Click += new System.EventHandler(this.IncBonus_Click);
            // 
            // Game
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.IncBonus);
            this.Controls.Add(this.Inc);
            this.Controls.Add(this.Add);
            this.Controls.Add(this.Points);
            this.Name = "Game";
            this.Text = "Form1";
            this.Load += new System.EventHandler(this.F_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label Points;
        private System.Windows.Forms.Button Add;
        private System.Windows.Forms.Label Inc;
        private System.Windows.Forms.Button IncBonus;
    }
}

