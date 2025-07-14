import React from 'react';

interface CountDisplayProps {
    balls: number;
    strikes: number;
    isComplete?: boolean;
    outcome?: string;
}

const CountDisplay: React.FC<CountDisplayProps> = ({ balls, strikes, isComplete, outcome }) => {
    const getCountColor = () => {
        if (isComplete) return 'text-gray-600';
        if (strikes >= 2) return 'text-red-600';
        if (balls >= 3) return 'text-blue-600';
        return 'text-gray-900';
    };

    const getOutcomeColor = () => {
        switch (outcome) {
            case 'strikeout':
                return 'text-red-600';
            case 'walk':
                return 'text-blue-600';
            case 'hit':
                return 'text-green-600';
            default:
                return 'text-gray-600';
        }
    };

    return (
        <div className="flex flex-col items-center p-4 bg-white rounded-lg shadow-md">
            <div className="text-sm font-medium text-gray-600 mb-2">COUNT</div>

            <div className={`text-4xl font-bold ${getCountColor()} mb-2`}>
                {balls}-{strikes}
            </div>

            {isComplete && outcome && (
                <div className={`text-lg font-semibold ${getOutcomeColor()}`}>
                    {outcome.toUpperCase()}
                </div>
            )}

            <div className="flex gap-4 mt-2">
                <div className="flex flex-col items-center">
                    <div className="text-xs text-gray-500">BALLS</div>
                    <div className="flex gap-1">
                        {[0, 1, 2, 3].map((i) => (
                            <div
                                key={i}
                                className={`w-3 h-3 rounded-full border-2 ${i < balls ? 'bg-blue-500 border-blue-500' : 'border-gray-300'
                                    }`}
                            />
                        ))}
                    </div>
                </div>

                <div className="flex flex-col items-center">
                    <div className="text-xs text-gray-500">STRIKES</div>
                    <div className="flex gap-1">
                        {[0, 1, 2].map((i) => (
                            <div
                                key={i}
                                className={`w-3 h-3 rounded-full border-2 ${i < strikes ? 'bg-red-500 border-red-500' : 'border-gray-300'
                                    }`}
                            />
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CountDisplay; 