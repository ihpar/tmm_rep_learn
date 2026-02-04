class FileParser:
    def __init__(self):
        self.note_line_codes = ["1", "4", "7", "8", "9",
                                "10", "11", "12", "23", "24", "28", "44"]

        self.pitches = ["G", "A", "B", "C", "D", "E", "F"]
        self.kommas = [9, 9, 4, 9, 9, 4, 9]

    def convert_to_sharp(self, note):
        """
        Converts notes with accidental symbols to all sharp rep."s

        G --> G, 
        Bb9 --> A, 
        Eb4 --> D#5, 
        Bb1 --> A#8, 
        Bb13 --> G#5, 
        Gb5 --> F#4, 
        F#4 --> F#4, 
        E#4 --> F, 
        E#5 --> F#1, 
        E#13 --> G
        """

        if len(note) == 1:
            return note

        accidental = note[1]
        if accidental not in ["#", "b"]:
            raise ValueError("Accidental symbol unrecognised!")

        root = note[0]
        idx = self.pitches.index(root)
        komma = int(note[2:])
        next_root, result = None, None
        remaining_komma = komma

        if accidental == "b":
            while remaining_komma > 0:
                idx = (idx - 1) % 7
                next_root = self.pitches[idx]
                remaining_komma -= self.kommas[idx]

            result = next_root
            if remaining_komma < 0:
                result += "#" + str(-1 * remaining_komma)

        elif accidental == "#":
            while remaining_komma > 0:
                remaining_komma -= self.kommas[idx]
                idx = (idx + 1) % 7
                next_root = self.pitches[idx]

            if remaining_komma < 0:
                idx = (idx - 1) % 7
                next_root = self.pitches[idx]
                remaining_komma += self.kommas[idx]

            result = next_root
            if remaining_komma != 0:
                result += "#" + str(remaining_komma)

        return result

    def parse_pitch(self, pitch_str):
        """
        pitch_str: D5, Es, E5b4, F5#4, etc.
        returns: PC, Acc, Oct
        """
        pc = "Es"
        acc = None
        oct = None
        if pitch_str != "Es":
            oct = int(pitch_str[1])
            sharp_repr = self.convert_to_sharp(pitch_str[0] + pitch_str[2:])
            pc = sharp_repr[0]
            if len(sharp_repr) > 1:
                acc = sharp_repr[1:]
        return pc, acc, oct

    def parse_line(self, tokens):
        """
        line: 60	9	Si4b2	B4b1	312	313	1	16	179	99	96		6.75
        returns: PC, Acc, Oct, Dur
        ex: ["A", "#8", 4, "1/16"]
        """
        if tokens[1] in self.note_line_codes:
            note = tokens[3]  # Some examples: D5, Es, E5b4, F5#4
            pc, acc, oct = self.parse_pitch(note)
            dur = tokens[6] + "/" + tokens[7]
            return [pc, acc, oct, dur]
        else:
            return None

    def parse_file(self, file_path):
        """
        returns list[[PC, Acc, Oct, Dur, MeasureBoundry]]
        """
        lines = None
        with open(file_path, "r") as in_file:
            lines = in_file.readlines()
        notes = []
        prev = "end"
        meas = None  # start || middle || end
        for line in lines:
            tokens = line.split("\t")
            line_result = self.parse_line(tokens)
            if line_result is None:
                continue

            if prev == "end":
                meas = "start"
            elif prev == "start" or prev == "middle":
                meas = "middle"
            if float(tokens[-1]) % 1 == 0:
                meas = "end"
            line_result.append(meas)
            notes.append(line_result)
            prev = meas

        return notes
