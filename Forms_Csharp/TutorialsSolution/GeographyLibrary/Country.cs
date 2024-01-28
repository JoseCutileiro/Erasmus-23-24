namespace GeographyLibrary
{
    public class Country
    {
        private string name;
        private int population;

        public Country(string name, int population)
        {
            this.name = name;
            this.population = population;
        }

        public Country()
        {
            return;
        }

        public string Name
        {
            get { return name; }
            set { name = value; }

        }

        public int Population
        {
            get { return population; }
            set { population = value; }
        }
    }
}