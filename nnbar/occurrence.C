
void occurrence()
{
  // seed data
  int array[] = { 1, 2, 3, 4, 3, 4, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 3 };
  const int size = sizeof( array ) / sizeof( array[0] );

  // create and fill the map
  std::map< int, int > occurances;
  for ( int i = 0; i < size; ++i )
    ++occurances[array[i]];

  // std::cout << "Greatest: " << occurances.rbegin()->second << '\n';
  // print the contents of the map
  using iterator = std::map<int, int>::iterator;
  for (iterator iter = occurances.begin(); iter != occurances.end(); ++iter) {
    std::cout << iter->first << ": " << iter->second << '\n';
  }
}
