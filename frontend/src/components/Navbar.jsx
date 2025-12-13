import { useState, useEffect, useRef } from "react";
import { Link } from "react-router-dom";

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <nav className="relative flex flex-wrap bg-opacity-30 items-center justify-between bg-zinc-900/30 py-2 shadow-sm dark:bg-neutral-700/30 lg:py-3">
      <div className="flex items-center justify-between px-5">
        <Link
          to="/#"
          className="flex items-center text-2xl ms-2 text-black/90 dark:text-white/60 font-semibold no-underline"
        >
          <img src="/horse.png" alt="logo" className="h-12 me-5" />
          <span className="pt-1">AI Thoroughbred Analytics</span>
        </Link>
      </div>

      <div className="flex items-center justify-between px-5">
        <div className="relative ms-2" ref={dropdownRef}>
          <button
            className="flex items-center px-6 pb-2 pt-1 text-black/80 transition duration-200 hover:text-black/80 focus:text-black/80 focus:outline-none dark:text-white/60 dark:hover:text-white/80 dark:focus:text-white/80 dark:active:text-white/80"
            onClick={() => setIsOpen(!isOpen)}
            type="button"
            id="dropdownMenuButton1"
            aria-expanded={isOpen}
          >
            Predictions
            <span className="ms-2 [&>svg]:w-5">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z"
                  clipRule="evenodd"
                />
              </svg>
            </span>
          </button>

          <ul
            className={`absolute z-[1000] right-0 mt-2 m-0 min-w-max list-none overflow-hidden rounded-lg border-none bg-zinc-100 dark:bg-neutral-700 bg-clip-padding text-left text-base shadow-lg transform origin-top-right transition-all duration-200 ease-out ${
              isOpen
                ? "scale-y-100 opacity-100"
                : "scale-y-0 opacity-0 pointer-events-none"
            }`}
            aria-labelledby="dropdownMenuButton1"
          >
            <li>
              <Link
                className="block w-full whitespace-nowrap px-4 py-2 text-sm font-normal text-neutral-700 hover:bg-zinc-200/60 focus:bg-zinc-200/60 focus:outline-none active:bg-zinc-200/60 dark:text-white dark:hover:bg-neutral-800/25 dark:focus:bg-neutral-800/25 dark:active:bg-neutral-800/25"
                to="/Predict"
                onClick={() => setIsOpen(false)}
              >
                Predict New Horse
              </Link>
            </li>
            <li>
              <Link
                className="block w-full whitespace-nowrap px-4 py-2 text-sm font-normal text-neutral-700 hover:bg-zinc-200/60 focus:bg-zinc-200/60 focus:outline-none active:bg-zinc-200/60 dark:text-white dark:hover:bg-neutral-800/25 dark:focus:bg-neutral-800/25 dark:active:bg-neutral-800/25"
                to="/Recall"
                onClick={() => setIsOpen(false)}
              >
                Recall Past Predictions
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
