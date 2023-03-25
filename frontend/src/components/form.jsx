import React, { useEffect, useState } from "react";
import axios from "axios";

function App() {
  const [joke, setJoke] = useState("");
  const [count, setCount] = useState(0);

  const  getJoke = async () => {
   await axios
      .get("https://icanhazdadjoke.com/", {
        headers: {
          Accept: "application/json",
        },
      })
      .then((response) => {
        setJoke(response.data.joke);
        setCount(count + 1);
      })
      .catch((error) => {
        console.log(error);
      });
  };

  useEffect(() => {
    getJoke()
  }, [])
  

  return (
    <div>
      <h1>Dad Joke</h1>
      <p>{joke}</p>
      <p>Number of new jokes: {count}</p>
      <button onClick={getJoke}>Get Joke</button>
    </div>
  );
}

export default App;
