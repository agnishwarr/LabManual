const displayCurrentDayTime = () => {
    const now = new Date();
    const days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    const day = days[now.getDay()];

    let hours = now.getHours();
    const minutes = now.getMinutes();
    const seconds = now.getSeconds();

    const ampm = hours >= 12 ? "PM" : "AM";

    hours = hours % 12;
    hours = hours ? hours : 12; // the hour '0' should be '12'
  
    // Display
    console.log("Today is : " + day + ".");
    console.log(`Current time is : ${hours} ${ampm} : ${minutes} : ${seconds}`);
  };
  
  // Run the function
  displayCurrentDayTime();
  