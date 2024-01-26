namespace Tutorial1Application
{
    partial class Tutorial1Form
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.mainMenuStrip = new System.Windows.Forms.MenuStrip();
            this.fileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.exitToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.setBlueButton = new System.Windows.Forms.Button();
            this.SetGreenButton = new System.Windows.Forms.Button();
            this.setRandButton = new System.Windows.Forms.Button();
            this.mainMenuStrip.SuspendLayout();
            this.SuspendLayout();
            // 
            // mainMenuStrip
            // 
            this.mainMenuStrip.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.mainMenuStrip.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.fileToolStripMenuItem});
            this.mainMenuStrip.Location = new System.Drawing.Point(0, 0);
            this.mainMenuStrip.Name = "mainMenuStrip";
            this.mainMenuStrip.Size = new System.Drawing.Size(379, 28);
            this.mainMenuStrip.TabIndex = 1;
            this.mainMenuStrip.Text = "menuStrip1";
            this.mainMenuStrip.ItemClicked += new System.Windows.Forms.ToolStripItemClickedEventHandler(this.menuStrip_ItemClicked);
            // 
            // fileToolStripMenuItem
            // 
            this.fileToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.exitToolStripMenuItem});
            this.fileToolStripMenuItem.Name = "fileToolStripMenuItem";
            this.fileToolStripMenuItem.Size = new System.Drawing.Size(46, 24);
            this.fileToolStripMenuItem.Text = "File";
            this.fileToolStripMenuItem.Click += new System.EventHandler(this.fileToolStripMenuItem_Click);
            // 
            // exitToolStripMenuItem
            // 
            this.exitToolStripMenuItem.Name = "exitToolStripMenuItem";
            this.exitToolStripMenuItem.Size = new System.Drawing.Size(224, 26);
            this.exitToolStripMenuItem.Text = "Exit";
            this.exitToolStripMenuItem.Click += new System.EventHandler(this.exitToolStripMenuItem_Click);
            // 
            // setBlueButton
            // 
            this.setBlueButton.Location = new System.Drawing.Point(36, 95);
            this.setBlueButton.Name = "setBlueButton";
            this.setBlueButton.Size = new System.Drawing.Size(75, 23);
            this.setBlueButton.TabIndex = 2;
            this.setBlueButton.Text = "Blue";
            this.setBlueButton.UseVisualStyleBackColor = true;
            this.setBlueButton.Click += new System.EventHandler(this.setBlueButton_Click);
            // 
            // SetGreenButton
            // 
            this.SetGreenButton.Location = new System.Drawing.Point(260, 95);
            this.SetGreenButton.Name = "SetGreenButton";
            this.SetGreenButton.Size = new System.Drawing.Size(75, 23);
            this.SetGreenButton.TabIndex = 3;
            this.SetGreenButton.Text = "Green";
            this.SetGreenButton.UseVisualStyleBackColor = true;
            this.SetGreenButton.Click += new System.EventHandler(this.SetGreenButton_Click);
            // 
            // setRandButton
            // 
            this.setRandButton.Location = new System.Drawing.Point(158, 233);
            this.setRandButton.Name = "setRandButton";
            this.setRandButton.Size = new System.Drawing.Size(75, 23);
            this.setRandButton.TabIndex = 4;
            this.setRandButton.Text = "Rand";
            this.setRandButton.UseVisualStyleBackColor = true;
            // 
            // Tutorial1Form
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(379, 322);
            this.Controls.Add(this.setRandButton);
            this.Controls.Add(this.SetGreenButton);
            this.Controls.Add(this.setBlueButton);
            this.Controls.Add(this.mainMenuStrip);
            this.Margin = new System.Windows.Forms.Padding(4);
            this.Name = "Tutorial1Form";
            this.Text = "Tutorial 1";
            this.Load += new System.EventHandler(this.Tutorial1Form_Load);
            this.mainMenuStrip.ResumeLayout(false);
            this.mainMenuStrip.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.MenuStrip mainMenuStrip;
        private System.Windows.Forms.ToolStripMenuItem fileToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem exitToolStripMenuItem;
        private System.Windows.Forms.Button setBlueButton;
        private System.Windows.Forms.Button SetGreenButton;
        private System.Windows.Forms.Button setRandButton;
    }
}

