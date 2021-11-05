# WiremindTest

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]





  <h3 align="center">Automated Fraud Detector</h3>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project was carried out for the admission process of Wiremind's junior data scientist position. The goal is to predict the unconstraint demand based on features among which come the price and the remaining days until departures

-Many models have been experimented (random Forest, SGBBOOST, 1DCONV) but I eventually chose the dense neural network because it converges well with the data provided

-The loss chosen is the MSE

-The feature selection have been carried out by evaluating the importance of each with statistical correlation methods (checkout the data analysis)

- The results of training and validation are logged in the results folder, where you can check the eval.csv file which contains  the table split per origin/destination, with the
average absolute for cumulated demand across all the timeline  [90, -60, -30, -20,-15, -10, -7, -6, -5, -3, -2, -1] and not just the final day 0.

-You can plot the evolution of cumulated true demand and cumulated predicted(checkout run.py file) for a particular train. The plot is saved in results folder.

-The relative error has not been used because of values equal to 0 given a very high error. The MAE has been used instead.


### Built With

This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [Docker](https://docs.docker.com/)




<!-- GETTING STARTED -->
## Getting Started

### Prerequisites


### Installation

1. Install Docker Desktop
2. Clone the repo
   ```sh
   git clone https://github.com/OmarKhatib96/WiremindTest.git
   ```
3. Build the docker image
   ```sh
   docker build -t wiremind .
   ```
4. Run the image project
 ```sh
   docker run wiremind
   ```
5. If the building/running of the image doesn't succeed and/or the results of the training/validation are not present in the result folder , run the project by executing the run.py file directly without docker.
   
 

<!-- USAGE EXAMPLES -->
## Usage


1. Modify the run.py file to check all the model functionalities (train, predict_forecast(price,day_x...), validation...)
  
2. Don't forget to rebuild the image if you modify the source code
3. If the rebuild doesn't succed , just run the run.py file
4. Check out the results of training and validation in the results folder.
   
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

*Omar KHATIB  - omar.khatib@dauphine.eu

Project Link: [https://github.com/OmarKhatib96/Fraud-Detection-Kafka](https://github.com/OmarKhatib96/WiremindTest)






<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]:https://www.linkedin.com/in/omar-khatib-b0758b12b/
[product-screenshot]: images/screenshot.png
